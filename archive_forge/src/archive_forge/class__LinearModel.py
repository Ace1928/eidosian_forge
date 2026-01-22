import abc
import collections
import math
import numpy as np
import six
from tensorflow.python.eager import context
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import template
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
class _LinearModel(base.Layer):
    """Creates a linear model using feature columns.

  See `linear_model` for details.
  """

    def __init__(self, feature_columns, units=1, sparse_combiner='sum', weight_collections=None, trainable=True, name=None, **kwargs):
        super(_LinearModel, self).__init__(name=name, **kwargs)
        self._keras_style = True
        self._feature_columns = _normalize_feature_columns(feature_columns)
        self._weight_collections = list(weight_collections or [])
        if ops.GraphKeys.GLOBAL_VARIABLES not in self._weight_collections:
            self._weight_collections.append(ops.GraphKeys.GLOBAL_VARIABLES)
        if ops.GraphKeys.MODEL_VARIABLES not in self._weight_collections:
            self._weight_collections.append(ops.GraphKeys.MODEL_VARIABLES)
        column_layers = {}
        for column in sorted(self._feature_columns, key=lambda x: x.name):
            with variable_scope.variable_scope(None, default_name=column._var_scope_name) as vs:
                column_name = _strip_leading_slashes(vs.name)
            column_layer = _FCLinearWrapper(column, units, sparse_combiner, self._weight_collections, trainable, column_name, **kwargs)
            column_layers[column_name] = column_layer
        self._column_layers = self._add_layers(column_layers)
        self._bias_layer = _BiasLayer(units=units, trainable=trainable, weight_collections=self._weight_collections, name='bias_layer', **kwargs)
        self._cols_to_vars = {}

    def cols_to_vars(self):
        """Returns a dict mapping _FeatureColumns to variables.

    See `linear_model` for more information.
    This is not populated till `call` is called i.e. layer is built.
    """
        return self._cols_to_vars

    def call(self, features):
        with variable_scope.variable_scope(self.name):
            for column in self._feature_columns:
                if not isinstance(column, (_DenseColumn, _CategoricalColumn)):
                    raise ValueError('Items of feature_columns must be either a _DenseColumn or _CategoricalColumn. Given: {}'.format(column))
            weighted_sums = []
            ordered_columns = []
            builder = _LazyBuilder(features)
            for layer in sorted(self._column_layers.values(), key=lambda x: x.name):
                column = layer._feature_column
                ordered_columns.append(column)
                weighted_sum = layer(builder)
                weighted_sums.append(weighted_sum)
                self._cols_to_vars[column] = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES, scope=layer.scope_name)
            _verify_static_batch_size_equality(weighted_sums, ordered_columns)
            predictions_no_bias = math_ops.add_n(weighted_sums, name='weighted_sum_no_bias')
            predictions = nn_ops.bias_add(predictions_no_bias, self._bias_layer(builder, scope=variable_scope.get_variable_scope()), name='weighted_sum')
            bias = self._bias_layer.variables[0]
            self._cols_to_vars['bias'] = _get_expanded_variable_list(bias)
        return predictions

    def _add_layers(self, layers):
        for name, layer in layers.items():
            setattr(self, 'layer-%s' % name, layer)
        return layers