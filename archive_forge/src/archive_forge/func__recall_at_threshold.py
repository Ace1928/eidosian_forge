from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _recall_at_threshold(labels, predictions, weights, threshold, name=None):
    with ops.name_scope(name, 'recall_at_%s' % threshold, (predictions, labels, weights, threshold)) as scope:
        precision_tensor, update_op = tf.compat.v1.metrics.recall_at_thresholds(labels=labels, predictions=predictions, thresholds=(threshold,), weights=weights, name=scope)
        return (tf.compat.v1.squeeze(precision_tensor), tf.compat.v1.squeeze(update_op))