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
def _eval_metric_ops(self, predicted_value, labels, weights, unreduced_loss, regularization_loss):
    """Returns the Eval metric ops."""
    keys = metric_keys.MetricKeys
    eval_metric_ops = {_summary_key(self._name, keys.LOSS_MEAN): tf.compat.v1.metrics.mean(values=unreduced_loss, weights=weights), _summary_key(self._name, keys.PREDICTION_MEAN): _predictions_mean(predictions=predicted_value, weights=weights, name=keys.PREDICTION_MEAN), _summary_key(self._name, keys.LABEL_MEAN): tf.compat.v1.metrics.mean(values=labels, weights=weights)}
    if regularization_loss is not None:
        regularization_loss_key = _summary_key(self._name, keys.LOSS_REGULARIZATION)
        eval_metric_ops[regularization_loss_key] = tf.compat.v1.metrics.mean(values=regularization_loss, name=keys.LOSS_REGULARIZATION)
    return eval_metric_ops