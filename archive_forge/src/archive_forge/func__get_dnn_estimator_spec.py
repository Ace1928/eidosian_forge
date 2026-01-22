from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _get_dnn_estimator_spec(use_tpu, head, features, labels, mode, logits, optimizer):
    """Get EstimatorSpec for DNN Model."""
    if use_tpu:
        return head._create_tpu_estimator_spec(features=features, mode=mode, labels=labels, optimizer=optimizer, logits=logits)
    else:
        return head.create_estimator_spec(features=features, mode=mode, labels=labels, optimizer=optimizer, logits=logits)