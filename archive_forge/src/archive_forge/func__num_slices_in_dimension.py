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
def _num_slices_in_dimension(self, axis: int) -> Optional[int]:
    """The total size of a dimension (like nvals).

      This is a static version of DynamicRaggedShape._num_slices_in_dimension()

      Example:

      ```
      shape = DynamicRaggedShape.Spec(
        _row_partitions=[
          RowPartitionSpec(nrows=3, nvals=14, dtype=tf.int32)
          RowPartitionSpec(nrows=14, nvals=25, dtype=tf.int32)

        ],
        _static_inner_shape=tf.TensorShape([25, 3, 4]),
        _inner_shape=tf.TensorSpec(tf.TensorShape([3]), dtype=tf.int32))
      shape._num_slices_in_dimension(0) = 3
      shape._num_slices_in_dimension(1) = 14
      shape._num_slices_in_dimension(2) = 25
      shape._num_slices_in_dimension(3) = 3
      shape._num_slices_in_dimension(4) = 4
      shape._num_slices_in_dimension(-2) = 3
      ```

      Args:
        axis: the last dimension to include.

      Returns:
        the number of values in a dimension.
      """
    if not isinstance(axis, int):
        raise TypeError('axis must be an integer')
    axis = array_ops.get_positive_axis(axis, self.rank, ndims_name='rank')
    if axis == 0:
        return self._dimension(0)
    if axis <= self.num_row_partitions:
        return self._row_partitions[axis - 1].nvals
    remainder = axis - (self.num_row_partitions - 1)
    head_inner_shape = self._static_inner_shape[:remainder]
    return head_inner_shape.num_elements()