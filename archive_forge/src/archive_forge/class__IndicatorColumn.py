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
class _IndicatorColumn(_DenseColumn, _SequenceDenseColumn, collections.namedtuple('_IndicatorColumn', ['categorical_column'])):
    """Represents a one-hot column for use in deep networks.

  Args:
    categorical_column: A `_CategoricalColumn` which is created by
      `categorical_column_with_*` function.
  """

    @property
    def name(self):
        return '{}_indicator'.format(self.categorical_column.name)

    def _transform_feature(self, inputs):
        """Returns dense `Tensor` representing feature.

    Args:
      inputs: A `_LazyBuilder` object to access inputs.

    Returns:
      Transformed feature `Tensor`.

    Raises:
      ValueError: if input rank is not known at graph building time.
    """
        id_weight_pair = self.categorical_column._get_sparse_tensors(inputs)
        id_tensor = id_weight_pair.id_tensor
        weight_tensor = id_weight_pair.weight_tensor
        if weight_tensor is not None:
            weighted_column = sparse_ops.sparse_merge(sp_ids=id_tensor, sp_values=weight_tensor, vocab_size=int(self._variable_shape[-1]))
            weighted_column = sparse_ops.sparse_slice(weighted_column, [0, 0], weighted_column.dense_shape)
            return array_ops.scatter_nd(weighted_column.indices, weighted_column.values, weighted_column.dense_shape)
        dense_id_tensor = sparse_ops.sparse_tensor_to_dense(id_tensor, default_value=-1)
        one_hot_id_tensor = array_ops.one_hot(dense_id_tensor, depth=self._variable_shape[-1], on_value=1.0, off_value=0.0)
        return math_ops.reduce_sum(one_hot_id_tensor, axis=[-2])

    @property
    def _parse_example_spec(self):
        return self.categorical_column._parse_example_spec

    @property
    def _variable_shape(self):
        """Returns a `TensorShape` representing the shape of the dense `Tensor`."""
        return tensor_shape.TensorShape([1, self.categorical_column._num_buckets])

    def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        """Returns dense `Tensor` representing feature.

    Args:
      inputs: A `_LazyBuilder` object to access inputs.
      weight_collections: Unused `weight_collections` since no variables are
        created in this function.
      trainable: Unused `trainable` bool since no variables are created in this
        function.

    Returns:
      Dense `Tensor` created within `_transform_feature`.

    Raises:
      ValueError: If `categorical_column` is a `_SequenceCategoricalColumn`.
    """
        del weight_collections
        del trainable
        if isinstance(self.categorical_column, _SequenceCategoricalColumn):
            raise ValueError('In indicator_column: {}. categorical_column must not be of type _SequenceCategoricalColumn. Suggested fix A: If you wish to use input_layer, use a non-sequence categorical_column_with_*. Suggested fix B: If you wish to create sequence input, use sequence_input_layer instead of input_layer. Given (type {}): {}'.format(self.name, type(self.categorical_column), self.categorical_column))
        return inputs.get(self)

    def _get_sequence_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        del weight_collections
        del trainable
        if not isinstance(self.categorical_column, _SequenceCategoricalColumn):
            raise ValueError('In indicator_column: {}. categorical_column must be of type _SequenceCategoricalColumn to use sequence_input_layer. Suggested fix: Use one of sequence_categorical_column_with_*. Given (type {}): {}'.format(self.name, type(self.categorical_column), self.categorical_column))
        dense_tensor = inputs.get(self)
        sparse_tensors = self.categorical_column._get_sparse_tensors(inputs)
        sequence_length = fc_utils.sequence_length_from_sparse_tensor(sparse_tensors.id_tensor)
        return _SequenceDenseColumn.TensorSequenceLengthPair(dense_tensor=dense_tensor, sequence_length=sequence_length)