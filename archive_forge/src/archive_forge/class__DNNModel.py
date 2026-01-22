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
class _DNNModel(tf.keras.Model):
    """A DNN Model."""

    def __init__(self, units, hidden_units, feature_columns, activation_fn, dropout, input_layer_partitioner, batch_norm, name=None, **kwargs):
        super(_DNNModel, self).__init__(name=name, **kwargs)
        if feature_column_lib.is_feature_column_v2(feature_columns):
            self._input_layer = tf.compat.v1.keras.layers.DenseFeatures(feature_columns=feature_columns, name='input_layer')
        else:
            self._input_layer = feature_column.InputLayer(feature_columns=feature_columns, name='input_layer', create_scope_now=False)
        self._add_layer(self._input_layer, 'input_layer')
        self._dropout = dropout
        self._batch_norm = batch_norm
        self._hidden_layers = []
        self._dropout_layers = []
        self._batch_norm_layers = []
        self._hidden_layer_scope_names = []
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.compat.v1.variable_scope('hiddenlayer_%d' % layer_id) as hidden_layer_scope:
                hidden_layer = tf.compat.v1.layers.Dense(units=num_hidden_units, activation=activation_fn, kernel_initializer=tf.compat.v1.glorot_uniform_initializer(), name=hidden_layer_scope, _scope=hidden_layer_scope)
                self._add_layer(hidden_layer, hidden_layer_scope.name)
                self._hidden_layer_scope_names.append(hidden_layer_scope.name)
                self._hidden_layers.append(hidden_layer)
                if self._dropout is not None:
                    dropout_layer = tf.compat.v1.layers.Dropout(rate=self._dropout)
                    self._add_layer(dropout_layer, dropout_layer.name)
                    self._dropout_layers.append(dropout_layer)
                if self._batch_norm:
                    batch_norm_layer = tf.compat.v1.layers.BatchNormalization(momentum=0.999, trainable=True, name='batchnorm_%d' % layer_id, _scope='batchnorm_%d' % layer_id)
                    self._add_layer(batch_norm_layer, batch_norm_layer.name)
                    self._batch_norm_layers.append(batch_norm_layer)
        with tf.compat.v1.variable_scope('logits') as logits_scope:
            self._logits_layer = tf.compat.v1.layers.Dense(units=units, activation=None, kernel_initializer=tf.compat.v1.glorot_uniform_initializer(), name=logits_scope, _scope=logits_scope)
            self._add_layer(self._logits_layer, logits_scope.name)
            self._logits_scope_name = logits_scope.name
        self._input_layer_partitioner = input_layer_partitioner

    def call(self, features, mode):
        is_training = mode == ModeKeys.TRAIN
        with ops.name_scope(name=_get_previous_name_scope()):
            with tf.compat.v1.variable_scope('input_from_feature_columns', partitioner=self._input_layer_partitioner):
                try:
                    net = self._input_layer(features, training=is_training)
                except TypeError:
                    net = self._input_layer(features)
            for i in range(len(self._hidden_layers)):
                net = self._hidden_layers[i](net)
                if self._dropout is not None and is_training:
                    net = self._dropout_layers[i](net, training=True)
                if self._batch_norm:
                    net = self._batch_norm_layers[i](net, training=is_training)
                _add_hidden_layer_summary(net, self._hidden_layer_scope_names[i])
            logits = self._logits_layer(net)
            _add_hidden_layer_summary(logits, self._logits_scope_name)
            return logits

    def _add_layer(self, layer, layer_name):
        setattr(self, layer_name, layer)