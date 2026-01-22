import warnings
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.dtensor import utils
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from keras.src.utils import control_flow_util
from keras.src.utils import tf_utils
from tensorflow.python.ops.control_flow_ops import (
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import keras_export
def _assign_new_value(self, variable, value):
    with backend.name_scope('AssignNewValue') as scope:
        if tf.compat.v1.executing_eagerly_outside_functions():
            return variable.assign(value, name=scope)
        else:
            with tf.compat.v1.colocate_with(variable):
                return tf.compat.v1.assign(variable, value, name=scope)