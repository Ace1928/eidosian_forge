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
def _linear_predictions(self, examples):
    """Returns predictions of the form w*x.

    Args:
      examples: Examples to compute predictions on.
    """
    with name_scope('sdca/prediction'):
        batch_size = tf.compat.v1.shape(examples['example_ids'])[0]
        predictions = tf.zeros([batch_size])
        sparse_variables = self._convert_n_to_tensor(self._variables['sparse_features_weights'])
        for sfc, sv in zip(examples['sparse_features'], sparse_variables):
            unpadded_dot_product = tf.math.segment_sum(tf.math.multiply(tf.compat.v1.gather(sv, sfc.feature_indices), sfc.feature_values), sfc.example_indices)
            predictions += tf.compat.v1.pad(unpadded_dot_product, [[0, batch_size - tf.compat.v1.shape(unpadded_dot_product)[0]]])
        dense_features = self._convert_n_to_tensor(examples['dense_features'])
        dense_variables = self._convert_n_to_tensor(self._variables['dense_features_weights'])
        for i in range(len(dense_variables)):
            predictions += tf.compat.v1.squeeze(tf.linalg.matmul(dense_features[i], tf.compat.v1.expand_dims(dense_variables[i], -1)))
    return predictions