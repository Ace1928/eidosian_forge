from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import six
import tensorflow as tf
from tensorflow.python.saved_model import model_utils as export_utils
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _check_is_tensor(x, tensor_name):
    """Returns `x` if it is a `Tensor`, raises TypeError otherwise."""
    if not isinstance(x, tf.compat.v2.__internal__.types.Tensor):
        raise TypeError('{} must be Tensor, given: {}'.format(tensor_name, x))
    return x