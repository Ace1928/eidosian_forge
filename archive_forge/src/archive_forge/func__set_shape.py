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
def _set_shape(self, shape):
    """Updates the static shape of `self` to be `shape`.

    * If a dimension of `shape` has known rank, and is encoded via
      partitioning, then this will update the corresponding partition to
      define `_uniform_row_length` and `nrows`.
    * If a dimension of `shape` has a known rank, and is encoded as one
      of the `flat_values` dimensions, then `flat_values.set_shape()` will
      be used to update its shape.

    Warning: Using this method to assert an incorrect shape for a RaggedTensor
    (i.e., one that's not consistent with its actual shape) can cause
    segmentation faults and very difficult-to-diagnose behavior.  Only use this
    method if you are certain that the shape is correct.

    Args:
      shape: `tf.TensorShape` specifying the shape for this `RaggedTensor`.
    """
    shape = tensor_shape.as_shape(shape)
    if shape.rank is None:
        return
    shape = shape.as_list()
    if shape[0] is not None:
        self._row_partition._row_splits.set_shape(shape[0] + 1)
    dtype = self._row_partition.dtype
    for i, partition in enumerate(self._nested_row_partitions):
        size = shape[i + 1]
        if size is not None:
            if partition._uniform_row_length is not None:
                old_row_length = tensor_util.constant_value(partition._uniform_row_length)
                if old_row_length is not None:
                    if size == old_row_length:
                        continue
                    else:
                        raise ValueError(f'Inconsistent size for axis {i + 1}: {old_row_length} vs. {size}.')
            partition._uniform_row_length = ops.convert_to_tensor(size, dtype)
            if partition._nrows is None:
                partition._nrows = array_ops.size(partition._row_splits, out_type=dtype) - 1
    if hasattr(self.flat_values, 'set_shape'):
        flat_shape = tensor_shape.as_shape([None] + shape[self.ragged_rank + 1:])
        self.flat_values.set_shape(flat_shape)