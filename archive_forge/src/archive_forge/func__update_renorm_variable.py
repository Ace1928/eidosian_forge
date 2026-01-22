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
def _update_renorm_variable(var, value, inputs_size):
    """Updates a moving average and weight, returns the unbiased
            value."""
    value = tf.identity(value)

    def _do_update():
        """Updates the var, returns the updated value."""
        new_var = self._assign_moving_average(var, value, self.renorm_momentum, inputs_size)
        return new_var

    def _fake_update():
        return tf.identity(var)
    return control_flow_util.smart_cond(training, _do_update, _fake_update)