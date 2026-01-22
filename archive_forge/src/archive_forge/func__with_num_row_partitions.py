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
def _with_num_row_partitions(self, new_num_row_partitions: int) -> 'DynamicRaggedShape.Spec':
    """Change the number of row partitions in the spec."""
    rank = self.rank
    if rank is None:
        raise ValueError('Changing num_row_partitions with unknown rank unsupported')
    if new_num_row_partitions > max(rank - 1, 0):
        raise ValueError('Number of row partitions too large')
    if new_num_row_partitions < 0:
        raise ValueError('Number of row partitions negative')
    if self.num_row_partitions == new_num_row_partitions:
        return self
    elif self.num_row_partitions < new_num_row_partitions:
        rp_delta = new_num_row_partitions - self.num_row_partitions
        tail_shape = DynamicRaggedShape.Spec._from_tensor_shape(self._static_inner_shape, rp_delta, self.dtype)
        return DynamicRaggedShape.Spec(row_partitions=self._row_partitions + tail_shape._row_partitions, static_inner_shape=tail_shape._static_inner_shape, dtype=self.dtype)
    else:
        assert self.num_row_partitions > new_num_row_partitions
        new_row_partitions = self._row_partitions[:new_num_row_partitions]
        last_row_partition = new_row_partitions[-1]
        old_row_partitions = self._row_partitions[new_num_row_partitions:]
        new_static_inner_shape = tensor_shape.TensorShape([last_row_partition.nvals] + [x.uniform_row_length for x in old_row_partitions]) + self._static_inner_shape[1:]
        return DynamicRaggedShape.Spec(new_row_partitions, new_static_inner_shape, self.dtype)