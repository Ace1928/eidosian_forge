import abc
import collections
import math
import re
import numpy as np
import six
from tensorflow.python.data.experimental.ops import lookup_ops as data_lookup_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.eager import context
from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.feature_column import feature_column_v2_types as fc_types
from tensorflow.python.feature_column import serialization
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import data_structures
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@serialization.register_feature_column
class WeightedCategoricalColumn(CategoricalColumn, fc_old._CategoricalColumn, collections.namedtuple('WeightedCategoricalColumn', ('categorical_column', 'weight_feature_key', 'dtype'))):
    """See `weighted_categorical_column`."""

    @property
    def _is_v2_column(self):
        return isinstance(self.categorical_column, fc_types.FeatureColumn) and self.categorical_column._is_v2_column

    @property
    def name(self):
        """See `FeatureColumn` base class."""
        return '{}_weighted_by_{}'.format(self.categorical_column.name, self.weight_feature_key)

    @property
    def parse_example_spec(self):
        """See `FeatureColumn` base class."""
        config = self.categorical_column.parse_example_spec
        if self.weight_feature_key in config:
            raise ValueError('Parse config {} already exists for {}.'.format(config[self.weight_feature_key], self.weight_feature_key))
        config[self.weight_feature_key] = parsing_ops.VarLenFeature(self.dtype)
        return config

    @property
    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _parse_example_spec(self):
        config = self.categorical_column._parse_example_spec
        if self.weight_feature_key in config:
            raise ValueError('Parse config {} already exists for {}.'.format(config[self.weight_feature_key], self.weight_feature_key))
        config[self.weight_feature_key] = parsing_ops.VarLenFeature(self.dtype)
        return config

    @property
    def num_buckets(self):
        """See `DenseColumn` base class."""
        return self.categorical_column.num_buckets

    @property
    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _num_buckets(self):
        return self.categorical_column._num_buckets

    def _transform_weight_tensor(self, weight_tensor):
        if weight_tensor is None:
            raise ValueError('Missing weights {}.'.format(self.weight_feature_key))
        weight_tensor = sparse_tensor_lib.convert_to_tensor_or_sparse_tensor(weight_tensor)
        if self.dtype != weight_tensor.dtype.base_dtype:
            raise ValueError('Bad dtype, expected {}, but got {}.'.format(self.dtype, weight_tensor.dtype))
        if not isinstance(weight_tensor, sparse_tensor_lib.SparseTensor):
            weight_tensor = _to_sparse_input_and_drop_ignore_values(weight_tensor, ignore_value=0.0)
        if not weight_tensor.dtype.is_floating:
            weight_tensor = math_ops.cast(weight_tensor, dtypes.float32)
        return weight_tensor

    def transform_feature(self, transformation_cache, state_manager):
        """Applies weights to tensor generated from `categorical_column`'."""
        weight_tensor = transformation_cache.get(self.weight_feature_key, state_manager)
        sparse_weight_tensor = self._transform_weight_tensor(weight_tensor)
        sparse_categorical_tensor = _to_sparse_input_and_drop_ignore_values(transformation_cache.get(self.categorical_column, state_manager))
        return (sparse_categorical_tensor, sparse_weight_tensor)

    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _transform_feature(self, inputs):
        """Applies weights to tensor generated from `categorical_column`'."""
        weight_tensor = inputs.get(self.weight_feature_key)
        weight_tensor = self._transform_weight_tensor(weight_tensor)
        return (inputs.get(self.categorical_column), weight_tensor)

    def get_sparse_tensors(self, transformation_cache, state_manager):
        """See `CategoricalColumn` base class."""
        tensors = transformation_cache.get(self, state_manager)
        return CategoricalColumn.IdWeightPair(tensors[0], tensors[1])

    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _get_sparse_tensors(self, inputs, weight_collections=None, trainable=None):
        del weight_collections
        del trainable
        tensors = inputs.get(self)
        return CategoricalColumn.IdWeightPair(tensors[0], tensors[1])

    @property
    def parents(self):
        """See 'FeatureColumn` base class."""
        return [self.categorical_column, self.weight_feature_key]

    def get_config(self):
        """See 'FeatureColumn` base class."""
        from tensorflow.python.feature_column.serialization import serialize_feature_column
        config = dict(zip(self._fields, self))
        config['categorical_column'] = serialize_feature_column(self.categorical_column)
        config['dtype'] = self.dtype.name
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None, columns_by_name=None):
        """See 'FeatureColumn` base class."""
        from tensorflow.python.feature_column.serialization import deserialize_feature_column
        _check_config_keys(config, cls._fields)
        kwargs = _standardize_and_copy_config(config)
        kwargs['categorical_column'] = deserialize_feature_column(config['categorical_column'], custom_objects, columns_by_name)
        kwargs['dtype'] = dtypes.as_dtype(config['dtype'])
        return cls(**kwargs)