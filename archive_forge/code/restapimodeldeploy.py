from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/predict", methods=["POST"])
def predict():
    # Get the input data from the request
    data = request.json["data"]

    # Make predictions using the loaded model
    predictions = model.predict(data)

    # Return the predictions as a JSON response
    return jsonify({"predictions": predictions.tolist()})


if __name__ == "__main__":
    app.run(debug=True)
