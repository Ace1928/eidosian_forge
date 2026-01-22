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
def _get_batch_size_and_size_checks(features, weight_column_key):
    """Returns batch_size and size_checks."""
    size_checks = []
    batch_size = None
    for key, feature in features.items():
        if key == weight_column_key:
            continue
        first_dim = tf.compat.v1.shape(feature)[0]
        if batch_size is None:
            batch_size = first_dim
        else:
            size_checks.append(tf.compat.v1.debugging.assert_equal(batch_size, first_dim))
    return (size_checks, batch_size)