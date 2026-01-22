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
@estimator_export(v1=['estimator.DNNRegressor'])
class DNNRegressor(estimator.Estimator):
    __doc__ = DNNRegressorV2.__doc__.replace('SUM_OVER_BATCH_SIZE', 'SUM')

    def __init__(self, hidden_units, feature_columns, model_dir=None, label_dimension=1, weight_column=None, optimizer='Adagrad', activation_fn=tf.nn.relu, dropout=None, input_layer_partitioner=None, config=None, warm_start_from=None, loss_reduction=tf.compat.v1.losses.Reduction.SUM, batch_norm=False):
        head = head_lib._regression_head(label_dimension=label_dimension, weight_column=weight_column, loss_reduction=loss_reduction)
        estimator._canned_estimator_api_gauge.get_cell('Regressor').set('DNN')

        def _model_fn(features, labels, mode, config):
            """Call the defined shared _dnn_model_fn."""
            return _dnn_model_fn(features=features, labels=labels, mode=mode, head=head, hidden_units=hidden_units, feature_columns=tuple(feature_columns or []), optimizer=optimizer, activation_fn=activation_fn, dropout=dropout, input_layer_partitioner=input_layer_partitioner, config=config, batch_norm=batch_norm)
        super(DNNRegressor, self).__init__(model_fn=_model_fn, model_dir=model_dir, config=config, warm_start_from=warm_start_from)