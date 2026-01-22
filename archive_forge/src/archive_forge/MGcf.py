import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("model.h5")

# Create a quantization-aware version of the model
quantize_model = tf.keras.models.clone_model(model)
quantize_model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input(shape=model.input_shape[1:]),
        tf.keras.layers.Rescaling(1.0 / 255),
        quantize_model,
        tf.keras.layers.Activation("softmax"),
    ]
)

# Compile the quantization-aware model
quantize_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Fine-tune the quantization-aware model
quantize_model.fit(x_train, y_train, epochs=5)

# Convert the model to a quantized TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(quantize_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the quantized model
with open("quantized_model.tflite", "wb") as f:
    f.write(tflite_model)
