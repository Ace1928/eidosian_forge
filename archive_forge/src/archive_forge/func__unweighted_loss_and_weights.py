from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
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
def _unweighted_loss_and_weights(self, logits, processed_labels, features):
    """Computes loss spec."""
    if self._loss_fn:
        unweighted_loss = base_head.call_loss_fn(loss_fn=self._loss_fn, labels=processed_labels, logits=logits, features=features, expected_loss_dim=1)
    else:
        unweighted_loss = tf.compat.v1.losses.sigmoid_cross_entropy(multi_class_labels=processed_labels, logits=logits, reduction=tf.compat.v1.losses.Reduction.NONE)
        unweighted_loss = tf.math.reduce_mean(unweighted_loss, axis=-1, keepdims=True)
    weights = base_head.get_weights_and_check_match_logits(features=features, weight_column=self._weight_column, logits=logits)
    return (unweighted_loss, weights)