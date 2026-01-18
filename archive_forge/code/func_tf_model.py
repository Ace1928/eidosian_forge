import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu, has_tensorflow
from thinc.util import to_categorical
from ..util import check_input_converters, make_tempdir
@pytest.fixture
def tf_model(n_hidden, input_size):
    import tensorflow as tf
    tf_model = tf.keras.Sequential([tf.keras.layers.Dense(n_hidden, input_shape=(input_size,)), tf.keras.layers.LayerNormalization(), tf.keras.layers.Dense(n_hidden, activation='relu'), tf.keras.layers.LayerNormalization(), tf.keras.layers.Dense(10, activation='softmax')])
    return tf_model