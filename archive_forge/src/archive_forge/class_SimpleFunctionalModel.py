import numpy as np
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.distribute import model_collection_base
from keras.src.optimizers.legacy import gradient_descent
class SimpleFunctionalModel(model_collection_base.ModelAndInput):
    """A simple functional model and its inputs."""

    def get_model(self, **kwargs):
        output_name = 'output_1'
        x = keras.layers.Input(shape=(3,), dtype=tf.float32)
        y = keras.layers.Dense(5, dtype=tf.float32, name=output_name)(x)
        model = keras.Model(inputs=x, outputs=y)
        optimizer = gradient_descent.SGD(learning_rate=0.001)
        model.compile(loss='mse', metrics=['mae'], optimizer=optimizer)
        return model

    def get_data(self):
        return _get_data_for_simple_models()

    def get_batch_size(self):
        return _BATCH_SIZE