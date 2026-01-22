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
class EmbeddingColumn(DenseColumn, SequenceDenseColumn, fc_old._DenseColumn, fc_old._SequenceDenseColumn, collections.namedtuple('EmbeddingColumn', ('categorical_column', 'dimension', 'combiner', 'initializer', 'ckpt_to_load_from', 'tensor_name_in_ckpt', 'max_norm', 'trainable', 'use_safe_embedding_lookup'))):
    """See `embedding_column`."""

    def __new__(cls, categorical_column, dimension, combiner, initializer, ckpt_to_load_from, tensor_name_in_ckpt, max_norm, trainable, use_safe_embedding_lookup=True):
        return super(EmbeddingColumn, cls).__new__(cls, categorical_column=categorical_column, dimension=dimension, combiner=combiner, initializer=initializer, ckpt_to_load_from=ckpt_to_load_from, tensor_name_in_ckpt=tensor_name_in_ckpt, max_norm=max_norm, trainable=trainable, use_safe_embedding_lookup=use_safe_embedding_lookup)

    @property
    def _is_v2_column(self):
        return isinstance(self.categorical_column, fc_types.FeatureColumn) and self.categorical_column._is_v2_column

    @property
    def name(self):
        """See `FeatureColumn` base class."""
        return '{}_embedding'.format(self.categorical_column.name)

    @property
    def parse_example_spec(self):
        """See `FeatureColumn` base class."""
        return self.categorical_column.parse_example_spec

    @property
    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _parse_example_spec(self):
        return self.categorical_column._parse_example_spec

    def transform_feature(self, transformation_cache, state_manager):
        """Transforms underlying `categorical_column`."""
        return transformation_cache.get(self.categorical_column, state_manager)

    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _transform_feature(self, inputs):
        return inputs.get(self.categorical_column)

    @property
    def variable_shape(self):
        """See `DenseColumn` base class."""
        return tensor_shape.TensorShape([self.dimension])

    @property
    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _variable_shape(self):
        return self.variable_shape

    def create_state(self, state_manager):
        """Creates the embedding lookup variable."""
        default_num_buckets = self.categorical_column.num_buckets if self._is_v2_column else self.categorical_column._num_buckets
        num_buckets = getattr(self.categorical_column, 'num_buckets', default_num_buckets)
        embedding_shape = (num_buckets, self.dimension)
        state_manager.create_variable(self, name='embedding_weights', shape=embedding_shape, dtype=dtypes.float32, trainable=self.trainable, use_resource=True, initializer=self.initializer)

    def _get_dense_tensor_internal_helper(self, sparse_tensors, embedding_weights):
        sparse_ids = sparse_tensors.id_tensor
        sparse_weights = sparse_tensors.weight_tensor
        if self.ckpt_to_load_from is not None:
            to_restore = embedding_weights
            if isinstance(to_restore, variables.PartitionedVariable):
                to_restore = to_restore._get_variable_list()
            checkpoint_utils.init_from_checkpoint(self.ckpt_to_load_from, {self.tensor_name_in_ckpt: to_restore})
        sparse_id_rank = tensor_shape.dimension_value(sparse_ids.dense_shape.get_shape()[0])
        embedding_lookup_sparse = embedding_ops.safe_embedding_lookup_sparse
        if not self.use_safe_embedding_lookup and sparse_id_rank is not None and (sparse_id_rank <= 2):
            embedding_lookup_sparse = embedding_ops.embedding_lookup_sparse_v2
        return embedding_lookup_sparse(embedding_weights, sparse_ids, sparse_weights, combiner=self.combiner, name='%s_weights' % self.name, max_norm=self.max_norm)

    def _get_dense_tensor_internal(self, sparse_tensors, state_manager):
        """Private method that follows the signature of get_dense_tensor."""
        embedding_weights = state_manager.get_variable(self, name='embedding_weights')
        return self._get_dense_tensor_internal_helper(sparse_tensors, embedding_weights)

    def _old_get_dense_tensor_internal(self, sparse_tensors, weight_collections, trainable):
        """Private method that follows the signature of _get_dense_tensor."""
        embedding_shape = (self.categorical_column._num_buckets, self.dimension)
        if weight_collections and ops.GraphKeys.GLOBAL_VARIABLES not in weight_collections:
            weight_collections.append(ops.GraphKeys.GLOBAL_VARIABLES)
        embedding_weights = variable_scope.get_variable(name='embedding_weights', shape=embedding_shape, dtype=dtypes.float32, initializer=self.initializer, trainable=self.trainable and trainable, collections=weight_collections)
        return self._get_dense_tensor_internal_helper(sparse_tensors, embedding_weights)

    def get_dense_tensor(self, transformation_cache, state_manager):
        """Returns tensor after doing the embedding lookup.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Embedding lookup tensor.

    Raises:
      ValueError: `categorical_column` is SequenceCategoricalColumn.
    """
        if isinstance(self.categorical_column, SequenceCategoricalColumn):
            raise ValueError('In embedding_column: {}. categorical_column must not be of type SequenceCategoricalColumn. Suggested fix A: If you wish to use DenseFeatures, use a non-sequence categorical_column_with_*. Suggested fix B: If you wish to create sequence input, use SequenceFeatures instead of DenseFeatures. Given (type {}): {}'.format(self.name, type(self.categorical_column), self.categorical_column))
        sparse_tensors = self.categorical_column.get_sparse_tensors(transformation_cache, state_manager)
        return self._get_dense_tensor_internal(sparse_tensors, state_manager)

    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        if isinstance(self.categorical_column, (SequenceCategoricalColumn, fc_old._SequenceCategoricalColumn)):
            raise ValueError('In embedding_column: {}. categorical_column must not be of type _SequenceCategoricalColumn. Suggested fix A: If you wish to use DenseFeatures, use a non-sequence categorical_column_with_*. Suggested fix B: If you wish to create sequence input, use SequenceFeatures instead of DenseFeatures. Given (type {}): {}'.format(self.name, type(self.categorical_column), self.categorical_column))
        sparse_tensors = self.categorical_column._get_sparse_tensors(inputs, weight_collections, trainable)
        return self._old_get_dense_tensor_internal(sparse_tensors, weight_collections, trainable)

    def get_sequence_dense_tensor(self, transformation_cache, state_manager):
        """See `SequenceDenseColumn` base class."""
        if not isinstance(self.categorical_column, SequenceCategoricalColumn):
            raise ValueError('In embedding_column: {}. categorical_column must be of type SequenceCategoricalColumn to use SequenceFeatures. Suggested fix: Use one of sequence_categorical_column_with_*. Given (type {}): {}'.format(self.name, type(self.categorical_column), self.categorical_column))
        sparse_tensors = self.categorical_column.get_sparse_tensors(transformation_cache, state_manager)
        dense_tensor = self._get_dense_tensor_internal(sparse_tensors, state_manager)
        sequence_length = fc_utils.sequence_length_from_sparse_tensor(sparse_tensors.id_tensor)
        return SequenceDenseColumn.TensorSequenceLengthPair(dense_tensor=dense_tensor, sequence_length=sequence_length)

    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _get_sequence_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        if not isinstance(self.categorical_column, (SequenceCategoricalColumn, fc_old._SequenceCategoricalColumn)):
            raise ValueError('In embedding_column: {}. categorical_column must be of type SequenceCategoricalColumn to use SequenceFeatures. Suggested fix: Use one of sequence_categorical_column_with_*. Given (type {}): {}'.format(self.name, type(self.categorical_column), self.categorical_column))
        sparse_tensors = self.categorical_column._get_sparse_tensors(inputs)
        dense_tensor = self._old_get_dense_tensor_internal(sparse_tensors, weight_collections=weight_collections, trainable=trainable)
        sequence_length = fc_utils.sequence_length_from_sparse_tensor(sparse_tensors.id_tensor)
        return SequenceDenseColumn.TensorSequenceLengthPair(dense_tensor=dense_tensor, sequence_length=sequence_length)

    @property
    def parents(self):
        """See 'FeatureColumn` base class."""
        return [self.categorical_column]

    def get_config(self):
        """See 'FeatureColumn` base class."""
        config = dict(zip(self._fields, self))
        config['categorical_column'] = serialization.serialize_feature_column(self.categorical_column)
        config['initializer'] = serialization._serialize_keras_object(self.initializer)
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None, columns_by_name=None):
        """See 'FeatureColumn` base class."""
        if 'use_safe_embedding_lookup' not in config:
            config['use_safe_embedding_lookup'] = True
        _check_config_keys(config, cls._fields)
        kwargs = _standardize_and_copy_config(config)
        kwargs['categorical_column'] = serialization.deserialize_feature_column(config['categorical_column'], custom_objects, columns_by_name)
        all_initializers = dict(tf_inspect.getmembers(init_ops, tf_inspect.isclass))
        kwargs['initializer'] = serialization._deserialize_keras_object(config['initializer'], module_objects=all_initializers, custom_objects=custom_objects)
        return cls(**kwargs)