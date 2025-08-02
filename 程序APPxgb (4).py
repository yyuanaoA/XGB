import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
model = joblib.load('xgb.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "Glycohemoglobin": {"type": "numerical", "min": 4.000, "max": 15.400, "default": 6.900},
    "Glucose": {"type": "numerical", "min": 47.000, "max": 554.000, "default": 143.000},
    "BRI": {"type": "numerical", "min": 2.756, "max": 18.297, "default": 10.951},
    "TC": {"type": "numerical", "min": 76, "max": 428.000, "default": 157},
    "Hypertensiontime": {"type": "numerical", "min": 0, "max": 63.000, "default": 16.000},
    "NHHR": {"type": "numerical", "min": 75, "max": 427, "default": 156.000},
    "BMI": {"type": "numerical", "min": 24, "max": 75.700, "default": 45.100},
    "NMLR": {"type": "numerical", "min": 0.320, "max": 13.000, "default": 2.920},
    "SII": {"type": "numerical", "min": 41, "max": 3551.18, "default": 945.000},
    "HDLC": {"type": "numerical", "min": 5, "max": 122, "default": 49.000},
}

# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果
    text = f"Based on feature values, predicted possibility of Diabetes is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(0.5, 0.5, text, fontsize=16, ha='center', va='center',
            fontname='Times New Roman', transform=ax.transAxes)
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")
    plt.close()

    # ---- 计算 SHAP ----
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    if isinstance(shap_values, list):
        shap_values_for_plot = shap_values[predicted_class][0]
    else:
        shap_values_for_plot = shap_values[0]

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        base_value = expected_value[predicted_class]
    else:
        base_value = expected_value

    # ---- 绘制 force_plot ----
    shap.force_plot(
        base_value,
        shap_values_for_plot,
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
        show=False
    )

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    plt.close()  # 关闭图防止文件不完整
    st.image("shap_force_plot.png")
