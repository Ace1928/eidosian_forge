from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import six
import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import dnn
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import linear
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _init_dnn_linear_combined_estimator(head, linear_feature_columns, linear_optimizer, dnn_feature_columns, dnn_optimizer, dnn_hidden_units, dnn_activation_fn, dnn_dropout, input_layer_partitioner, linear_sparse_combiner):
    """Helper function for the initialization of DNNLinearCombinedEstimator."""
    linear_feature_columns = linear_feature_columns or []
    dnn_feature_columns = dnn_feature_columns or []
    feature_columns = list(linear_feature_columns) + list(dnn_feature_columns)
    if not feature_columns:
        raise ValueError('Either linear_feature_columns or dnn_feature_columns must be defined.')

    def _model_fn(features, labels, mode, config):
        """Call the _dnn_linear_combined_model_fn."""
        return _dnn_linear_combined_model_fn(features=features, labels=labels, mode=mode, head=head, linear_feature_columns=linear_feature_columns, linear_optimizer=linear_optimizer, dnn_feature_columns=dnn_feature_columns, dnn_optimizer=dnn_optimizer, dnn_hidden_units=dnn_hidden_units, dnn_activation_fn=dnn_activation_fn, dnn_dropout=dnn_dropout, input_layer_partitioner=input_layer_partitioner, config=config, linear_sparse_combiner=linear_sparse_combiner)
    return (feature_columns, _model_fn)