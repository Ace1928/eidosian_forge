import abc
from typing import Any, Iterable, Optional, Sequence, Tuple, Union
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.ragged.row_partition import RowPartitionSpec
from tensorflow.python.types import core
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
class _Broadcaster:
    """A _Broadcaster represents a transformation from one shape to another.

  It provides a transform for each axis of the source shape to the
  corresponding axis of the destination shape.

  """

    def __init__(self, source_shape, target_shape, layer_broadcasters, dtype=None):
        """Create a broadcaster.

    Do not call directly.
    The source_shape, target_shape, and layer_broadcasters are converted
    to have the same dtype.

    Note: source_shape.rank and target_shape.rank must be known.
    Args:
      source_shape: the source DynamicRaggedShape
      target_shape: the target DynamicRaggedShape
      layer_broadcasters: List[_LayerBroadcaster] of length source_shape.rank.
      dtype: the preferred dtype of the broadcaster.

    Raises:
      TypeError: if the input types don't match.
    """
        if not isinstance(source_shape, DynamicRaggedShape):
            raise TypeError('source_shape is not a DynamicRaggedShape')
        if not isinstance(target_shape, DynamicRaggedShape):
            raise TypeError('target_shape is not a DynamicRaggedShape')
        if not isinstance(layer_broadcasters, list):
            raise TypeError('layer_broadcasters not a list: ' + str(layer_broadcasters))
        for bc in layer_broadcasters:
            if not isinstance(bc, _LayerBroadcaster):
                raise TypeError('Not a LayerBroadcaster: ' + str(bc))
        dtype = _find_dtype(source_shape, dtype)
        dtype = _find_dtype(target_shape, dtype)
        dtype = _find_dtype_iterable(layer_broadcasters, dtype)
        dtype = _find_dtype(dtypes.int64, dtype)
        self._source_shape = source_shape.with_dtype(dtype)
        self._target_shape = target_shape.with_dtype(dtype)
        self._layer_broadcasters = [x.with_dtype(dtype) for x in layer_broadcasters]

    def __repr__(self):
        return '{src_shape:' + str(self._source_shape) + ', target_shape:' + str(self._target_shape) + ' layer_broadcasters: ' + str(self._layer_broadcasters) + '}'

    def with_dtype(self, dtype):
        """Return a copy of this Broadcaster with a different dtype."""
        return _Broadcaster(self._source_shape, self._target_shape, self._layer_broadcasters, dtype)

    @property
    def source_shape(self):
        return self._source_shape

    @property
    def target_shape(self):
        return self._target_shape

    @property
    def dtype(self):
        return self._source_shape.dtype

    def _target_inner_shape_int32(self):
        new_inner_shape = self.target_shape.inner_shape
        if new_inner_shape.dtype == dtypes.int64:
            new_inner_shape = math_ops.cast(new_inner_shape, dtype=dtypes.int32)
        return new_inner_shape

    def broadcast_flat_values(self, rt, inner_dimensions=True):
        """flat_values of a ragged tensor broadcast to target_shape.

    If inner_dimensions==True, then the result is a dense tensor with shape
    target_shape.inner_shape, the flat values of the broadcasted shape.

    If you add target_shape.row_partitions, you will get the full broadcasted
    shape.

    If inner_dimensions==False, the result is a dense tensor that satsifies
    certain properties:
    1. broadcast_to(result, target_shape.inner_shape) will give the result
       if inner_dimensions==True.
    2. Either (a) (result.rank < target_shape.inner_rank)
       or (b) (result.shape[0] == target_shape.inner_shape[0]).
    3. result.rank = min(target_shape.inner_rank, rt.rank)
    4. For i < target_shape.inner_rank - 1, and i < rt.rank,
       and if rt.shape[-i]!=1, then result.shape[-i]=target_shape[-i].
    Args:
      rt: a ragged or dense tensor.
      inner_dimensions: if true, broadcast the inner dimensions as well.

    Returns:
      a dense tensor
    """
        if ragged_tensor.is_ragged(rt):
            rt = rt.flat_values
        if self.target_shape.rank == 0:
            return rt
        inner_rank = self.target_shape.inner_rank
        if inner_rank > self._source_shape.rank:
            if self.source_shape.num_row_partitions > 0:
                rt = array_ops.reshape(rt, self.source_shape._alt_inner_shape(self.source_shape.rank))
            if inner_dimensions:
                return array_ops.broadcast_to(rt, self._target_inner_shape_int32())
            return rt
        else:
            if self._source_shape.inner_rank != inner_rank:
                rt = array_ops.reshape(rt, self._source_shape._alt_inner_shape(inner_rank))
            flat_broadcaster = self._layer_broadcasters[-inner_rank]
            rt = flat_broadcaster.broadcast_tensor(rt)
            if inner_dimensions:
                rt = array_ops.broadcast_to(rt, self._target_inner_shape_int32())
            return rt

    def broadcast(self, rt):
        """Broadcast a tensor of source_shape to target_shape."""
        flat_values = self.broadcast_flat_values(rt)
        return self.target_shape._add_row_partitions(flat_values)