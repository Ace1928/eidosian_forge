import functools
import operator
import typing
import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_config
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_types
from tensorflow.python.types import internal as internal_types
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@classmethod
def _from_row_partition(cls, values, row_partition, validate=True):
    """Creates a `RaggedTensor` with a row partition.

    This is used as a way for RaggedTensors to share row partitions.

    The outer dimension of values must be equal to `partition.nvals()`.

    Args:
      values: A potentially ragged tensor.
      row_partition: a `RowPartition`: can be shared between tensors.
      validate: If true, then use assertions to check that the arguments form a
        valid `RaggedTensor`.

    Returns:
      A `RaggedTensor`.  `result.rank = values.rank + 1`.
      `result.ragged_rank = values.ragged_rank + 1`.

    Raises:
      ValueError: If partition.nvals() != _nrows(values)
    """
    if not isinstance(row_partition, RowPartition):
        raise TypeError(f'Argument `row_partition` must be a RowPartition. Received {row_partition}.')
    if not isinstance(validate, bool):
        raise TypeError(f'Argument `validate` must have type bool. Received {validate}.')
    values, row_partition = cls._convert_values_and_partition(values, row_partition, 'partition')
    if row_partition._has_precomputed_value_rowids():
        value_rowids_shape = row_partition.value_rowids().shape
        values.shape[:1].assert_is_compatible_with(value_rowids_shape)
    if validate:
        msg = 'Arguments to _from_row_partition do not form a valid RaggedTensor'
        nvals = _nrows(values, row_partition.dtype)
        checks = [check_ops.assert_equal(math_ops.cast(row_partition.nvals(), row_partition.dtype), nvals, message=msg)]
        if not isinstance(values, RaggedTensor):
            checks.append(check_ops.assert_rank_at_least(values, 1))
        row_partition = row_partition._with_dependencies(checks)
    return cls(values=values, internal=True, row_partition=row_partition)