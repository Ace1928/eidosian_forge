from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import lookup_ops
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.head import base_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _update_auc(self, auc_metric, labels, predictions, weights=None):
    predictions = tf.cast(predictions, dtype=tf.dtypes.float32)
    if weights is not None:
        weights = tf.compat.v2.__internal__.ops.broadcast_weights(weights, predictions)
    auc_metric.update_state(y_true=labels, y_pred=predictions, sample_weight=weights)