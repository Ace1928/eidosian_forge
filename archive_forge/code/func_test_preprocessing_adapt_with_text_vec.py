from unittest import mock
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
import autokeras as ak
from autokeras import keras_layers
from autokeras import test_utils
from autokeras.engine import tuner as tuner_module
from autokeras.tuners import greedy
def test_preprocessing_adapt_with_text_vec():

    class MockLayer(preprocessing.TextVectorization):

        def adapt(self, *args, **kwargs):
            super().adapt(*args, **kwargs)
            self.is_called = True
    x_train = test_utils.generate_text_data()
    y_train = np.random.randint(0, 2, (100,))
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    layer1 = MockLayer(max_tokens=5000, output_mode='int', output_sequence_length=40)
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(1,), dtype=tf.string))
    model.add(layer1)
    model.add(keras.layers.Embedding(50001, 10))
    model.add(keras.layers.Dense(1))
    tuner_module.AutoTuner.adapt(model, dataset)
    assert layer1.is_called