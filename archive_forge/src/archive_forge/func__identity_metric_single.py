from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys
from tensorflow_estimator.python.estimator.export import export_lib
def _identity_metric_single(name, input_tensor):
    """A metric which takes on its last updated value.

  This keeps evaluation metrics in sync with one another, since update ops are
  run separately from their result Tensors. Simply returning (input_tensor,
  no_op) as a metric with a value but no update means that a metric will come
  from a different batch of data than metrics which cache values in a Variable
  (e.g. the default loss metric).

  Args:
    name: A name for the metric.
    input_tensor: Any Tensor.

  Returns:
    A tuple of (value, update_op).
  """
    metric_variable = tf.compat.v1.Variable(name='{}_identity_metric'.format(name), initial_value=tf.zeros([], dtype=input_tensor.dtype), collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES], validate_shape=False)
    update_op = tf.compat.v1.assign(metric_variable, input_tensor, validate_shape=False)
    metric_variable.set_shape(input_tensor.get_shape())
    return (metric_variable.value(), update_op)