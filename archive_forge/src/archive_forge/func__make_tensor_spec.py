import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src.engine import base_layer
from keras.src.engine import functional
from keras.src.engine import sequential
from keras.src.utils import io_utils
def _make_tensor_spec(x):
    return tf.TensorSpec(x.shape, dtype=x.dtype, name=x.name)