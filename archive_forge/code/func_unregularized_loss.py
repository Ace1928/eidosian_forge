from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
from six.moves import range
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework.ops import internal_convert_to_tensor
from tensorflow.python.framework.ops import name_scope
from tensorflow.python.ops import gen_sdca_ops
from tensorflow.python.ops import variables as var_ops
from tensorflow.python.ops.nn import log_poisson_loss
from tensorflow.python.ops.nn import sigmoid_cross_entropy_with_logits
from tensorflow_estimator.python.estimator.canned.linear_optimizer.python.utils.sharded_mutable_dense_hashtable import _ShardedMutableDenseHashTable
def unregularized_loss(self, examples):
    """Add operations to compute the loss (without the regularization loss).

    Args:
      examples: Examples to compute unregularized loss on.

    Returns:
      An Operation that computes mean (unregularized) loss for given set of
      examples.

    Raises:
      ValueError: if examples are not well defined.
    """
    self._assert_specified(['example_labels', 'example_weights', 'sparse_features', 'dense_features'], examples)
    self._assert_list(['sparse_features', 'dense_features'], examples)
    with name_scope('sdca/unregularized_loss'):
        predictions = tf.cast(self._linear_predictions(examples), tf.dtypes.float64)
        labels = tf.cast(internal_convert_to_tensor(examples['example_labels']), tf.dtypes.float64)
        weights = tf.cast(internal_convert_to_tensor(examples['example_weights']), tf.dtypes.float64)
        if self._options['loss_type'] == 'logistic_loss':
            return tf.math.reduce_sum(tf.math.multiply(sigmoid_cross_entropy_with_logits(labels=labels, logits=predictions), weights)) / tf.math.reduce_sum(weights)
        if self._options['loss_type'] == 'poisson_loss':
            return tf.math.reduce_sum(tf.math.multiply(log_poisson_loss(targets=labels, log_input=predictions), weights)) / tf.math.reduce_sum(weights)
        if self._options['loss_type'] in ['hinge_loss', 'smooth_hinge_loss']:
            all_ones = tf.compat.v1.ones_like(predictions)
            adjusted_labels = tf.math.subtract(2 * labels, all_ones)
            error = tf.nn.relu(tf.math.subtract(all_ones, tf.math.multiply(adjusted_labels, predictions)))
            weighted_error = tf.math.multiply(error, weights)
            return tf.math.reduce_sum(weighted_error) / tf.math.reduce_sum(weights)
        err = tf.math.subtract(labels, predictions)
        weighted_squared_err = tf.math.multiply(tf.math.square(err), weights)
        return tf.math.reduce_sum(weighted_squared_err) / (2.0 * tf.math.reduce_sum(weights))