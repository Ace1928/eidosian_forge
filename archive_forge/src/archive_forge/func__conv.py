import warnings
import tensorflow as tf
from keras.src.backend import standardize_data_format
from keras.src.backend import standardize_dtype
from keras.src.backend.common.backend_utils import (
from keras.src.backend.config import epsilon
from keras.src.backend.tensorflow.core import cast
from keras.src.backend.tensorflow.core import convert_to_tensor
def _conv():
    tf_data_format = _convert_data_format(data_format, len(inputs.shape))
    return tf.nn.convolution(inputs, kernel, strides, padding.upper(), data_format=tf_data_format, dilations=dilation_rate)