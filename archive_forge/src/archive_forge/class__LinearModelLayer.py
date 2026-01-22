from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.feature_column import feature_column_v2 as fc_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.canned.linear_optimizer.python.utils import sdca_ops
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import binary_class_head
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
class _LinearModelLayer(tf.keras.layers.Layer):
    """Layer that contains logic for `LinearModel`."""

    def __init__(self, feature_columns, units=1, sparse_combiner='sum', trainable=True, name=None, **kwargs):
        super(_LinearModelLayer, self).__init__(name=name, trainable=trainable, **kwargs)
        self._feature_columns = fc_v2._normalize_feature_columns(feature_columns)
        for column in self._feature_columns:
            if not isinstance(column, (tf.compat.v2.__internal__.feature_column.DenseColumn, fc_v2.CategoricalColumn)):
                raise ValueError('Items of feature_columns must be either a DenseColumn or CategoricalColumn. Given: {}'.format(column))
        self._units = units
        self._sparse_combiner = sparse_combiner
        self._state_manager = tf.compat.v2.__internal__.feature_column.StateManager(self, self.trainable)
        self.bias = None

    def build(self, _):
        with variable_scope._pure_variable_scope(self.name):
            for column in self._feature_columns:
                with variable_scope._pure_variable_scope(fc_v2._sanitize_column_name_for_variable_scope(column.name)):
                    column.create_state(self._state_manager)
                    if isinstance(column, fc_v2.CategoricalColumn):
                        first_dim = column.num_buckets
                    else:
                        first_dim = column.variable_shape.num_elements()
                    self._state_manager.create_variable(column, name='weights', dtype=tf.float32, shape=(first_dim, self._units), initializer=tf.keras.initializers.zeros(), trainable=self.trainable)
            self.bias = self.add_weight(name='bias_weights', dtype=tf.float32, shape=[self._units], initializer=tf.keras.initializers.zeros(), trainable=self.trainable, use_resource=True, getter=tf.compat.v1.get_variable)
        super(_LinearModelLayer, self).build(None)

    def call(self, features):
        if not isinstance(features, dict):
            raise ValueError('We expected a dictionary here. Instead we got: {}'.format(features))
        with ops.name_scope(self.name):
            transformation_cache = tf.compat.v2.__internal__.feature_column.FeatureTransformationCache(features)
            weighted_sums = []
            for column in self._feature_columns:
                with ops.name_scope(fc_v2._sanitize_column_name_for_variable_scope(column.name)):
                    weight_var = self._state_manager.get_variable(column, 'weights')
                    weighted_sum = fc_v2._create_weighted_sum(column=column, transformation_cache=transformation_cache, state_manager=self._state_manager, sparse_combiner=self._sparse_combiner, weight_var=weight_var)
                    weighted_sums.append(weighted_sum)
            fc_v2._verify_static_batch_size_equality(weighted_sums, self._feature_columns)
            predictions_no_bias = tf.math.add_n(weighted_sums, name='weighted_sum_no_bias')
            predictions = tf.nn.bias_add(predictions_no_bias, self.bias, name='weighted_sum')
            return predictions

    def get_config(self):
        from tensorflow.python.feature_column import serialization
        column_configs = serialization.serialize_feature_columns(self._feature_columns)
        config = {'feature_columns': column_configs, 'units': self._units, 'sparse_combiner': self._sparse_combiner}
        base_config = super(_LinearModelLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from tensorflow.python.feature_column import serialization
        config_cp = config.copy()
        columns = serialization.deserialize_feature_columns(config_cp['feature_columns'], custom_objects=custom_objects)
        del config_cp['feature_columns']
        return cls(feature_columns=columns, **config_cp)