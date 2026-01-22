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
def _slice_shape(self, start, stop):
    """Returns a shape self[start:stop].

    If start == 0, then this truncates dimensions after stop.
    If start != 0, then this will return a shape with num_row_partitions == 0.

    See __getitem__.

    Args:
      start: the first dimension. 0 <= start <= rank
      stop: the last dimension (exclusive). 0 <= stop <= rank
    """
    if stop <= start:
        return DynamicRaggedShape._from_inner_shape([])
    elif start == 0:
        if stop <= self.num_row_partitions:
            if stop == 1:
                return DynamicRaggedShape._from_inner_shape([self.row_partitions[0].nrows()])
            new_row_partitions = self.row_partitions[:stop - 1]
            new_inner_shape = [new_row_partitions[-1].nvals()]
            return DynamicRaggedShape(new_row_partitions, new_inner_shape)
        else:
            if self.rank is None:
                new_inner_rank = stop - self.num_row_partitions
                new_inner_shape = self.inner_shape[:new_inner_rank]
                return DynamicRaggedShape(row_partitions=self.row_partitions, inner_shape=new_inner_shape, static_inner_shape=None, validate=False)
            elif self.rank <= stop:
                return self
            new_inner_rank = stop - self.num_row_partitions
            new_inner_shape = self.inner_shape[:new_inner_rank]
            return DynamicRaggedShape(row_partitions=self.row_partitions, inner_shape=new_inner_shape, static_inner_shape=tensor_shape.TensorShape([None] * new_inner_rank), validate=False)
    else:
        if self.rank is None or stop < self.rank:
            partial = self._slice_shape(0, stop)
        else:
            partial = self
        for x in partial.row_partitions:
            if not x.is_uniform():
                raise ValueError('All relevant dimensions must be uniform')
        if partial.rank is None:
            raise NotImplementedError('__getitem__[start:stop] where start > 0 not implemented')
        return DynamicRaggedShape._from_inner_shape(partial._with_num_row_partitions(0).inner_shape[start:])