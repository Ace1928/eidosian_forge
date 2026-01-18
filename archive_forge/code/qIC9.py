from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier

# Load the dataset
X_train = ...  # Training features
y_train = ...  # Training labels
X_test = ...  # Testing features
y_test = ...  # Testing labels

# Define the base models
base_models = [
    ("lr", LogisticRegression()),
    ("dt", DecisionTreeClassifier()),
    ("rf", RandomForestClassifier()),
]

# Define the meta-model
meta_model = LogisticRegression()

# Create the stacking classifier
stacking_classifier = StackingClassifier(
    classifiers=base_models, meta_classifier=meta_model
)

# Train the stacking classifier
stacking_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = stacking_classifier.predict(X_test)

# Evaluate the stacking classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Stacking Classifier Accuracy: {accuracy:.3f}")
