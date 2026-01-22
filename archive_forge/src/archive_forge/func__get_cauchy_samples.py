import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import initializers
from keras.src.engine import base_layer
from keras.src.engine import input_spec
from tensorflow.python.util.tf_export import keras_export
def _get_cauchy_samples(loc, scale, shape):
    probs = np.random.uniform(low=0.0, high=1.0, size=shape)
    return loc + scale * np.tan(np.pi * (probs - 0.5))