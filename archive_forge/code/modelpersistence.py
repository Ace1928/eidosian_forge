import pickle

# Train a model
model = ...  # Trained model

# Save the model to disk
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

# Load the model from disk
with open("model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Make predictions using the loaded model
predictions = loaded_model.predict(...)
