import lime
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the dataset
X_train = ...  # Training text data
y_train = ...  # Training labels
X_test = ...  # Testing text data

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a logistic regression classifier
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train_vectorized, y_train)

# Create a LIME explainer
explainer = LimeTextExplainer(class_names=["Negative", "Positive"])

# Select an instance to explain
instance = X_test

# Generate the LIME explanation
exp = explainer.explain_instance(instance, lr_classifier.predict_proba, num_features=10)

# Visualize the LIME explanation
exp.show_in_notebook(text=instance)
