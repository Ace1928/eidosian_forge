from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v2 as tf
from keras.src.feature_column import base_feature_layer as kfc
from keras.src.feature_column import dense_features
from keras.src.utils import tf_contextlib
from tensorflow.python.util.tf_export import keras_export
class _StateManagerImplV2(tf.__internal__.feature_column.StateManager):
    """Manages the state of DenseFeatures."""

    def create_variable(self, feature_column, name, shape, dtype=None, trainable=True, use_resource=True, initializer=None):
        if name in self._cols_to_vars_map[feature_column]:
            raise ValueError('Variable already exists.')
        with no_manual_dependency_tracking_scope(self._layer):
            var = self._layer.add_weight(name=name, shape=shape, dtype=dtype, initializer=initializer, trainable=self._trainable and trainable, use_resource=use_resource)
        if isinstance(var, tf.__internal__.tracking.Trackable):
            self._layer._track_trackable(var, feature_column.name + '/' + name)
        self._cols_to_vars_map[feature_column][name] = var
        return var