import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from autokeras import keras_layers as layer_module
def test_transformer_encoder_save_and_load(tmp_path):
    layer = layer_module.BertEncoder()
    inputs = [keras.Input(shape=(500,), dtype=tf.int64), keras.Input(shape=(500,), dtype=tf.int64), keras.Input(shape=(500,), dtype=tf.int64)]
    model = keras.Model(inputs, layer(inputs))
    model.save(os.path.join(tmp_path, 'model'))
    keras.models.load_model(os.path.join(tmp_path, 'model'))