import shap
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
X = ...  # Features
y = ...  # Target labels

# Train a random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Create a SHAP explainer
explainer = shap.TreeExplainer(model)

# Generate SHAP values for a specific instance
instance = ...  # Instance to explain
shap_values = explainer.shap_values(instance)

# Visualize the SHAP values
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], instance)
