from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column as feature_column_v1
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _baseline_model_fn_builder_v2(features, num_outputs, weight_column=None):
    """Function builder for a baseline logit_fn.

  Args:
    features: The first item returned from the `input_fn` passed to `train`,
      `evaluate`, and `predict`. This should be a single `Tensor` or dict with
      `Tensor` values.
    num_outputs: Number of outputs for the model.
    weight_column: A string or a `NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It will be multiplied by the loss of the example.

  Returns:
    A list of trainable variables and a `Tensor` representing the logits.
  """
    weight_column_key = _get_weight_column_key_v2(weight_column)
    size_checks, batch_size = _get_batch_size_and_size_checks(features, weight_column_key)
    with tf.control_dependencies(size_checks):
        with ops.name_scope('baseline'):
            bias = tf.Variable(initial_value=tf.zeros([num_outputs]), name='bias')
            logits = tf.math.multiply(bias, tf.ones([batch_size, num_outputs]))
    return ([bias], logits)