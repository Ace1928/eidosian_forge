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
def _accuracy_at_threshold(labels, predictions, weights, threshold, name=None):
    with ops.name_scope(name, 'accuracy_at_%s' % threshold, (predictions, labels, weights, threshold)) as scope:
        threshold_predictions = tf.compat.v1.to_float(tf.math.greater_equal(predictions, threshold))
        return tf.compat.v1.metrics.accuracy(labels=labels, predictions=threshold_predictions, weights=weights, name=scope)