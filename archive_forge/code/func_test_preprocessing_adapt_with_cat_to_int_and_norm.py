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
def test_preprocessing_adapt_with_cat_to_int_and_norm():
    x = np.array([['a', 5], ['b', 6]]).astype(np.unicode)
    y = np.array([[1, 2], [3, 4]]).astype(np.unicode)
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(32)
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(2,), dtype=tf.string))
    model.add(keras_layers.MultiCategoryEncoding(['int', 'none']))
    model.add(preprocessing.Normalization(axis=-1))
    tuner_module.AutoTuner.adapt(model, dataset)