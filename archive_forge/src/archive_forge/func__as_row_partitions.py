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
def _as_row_partitions(self):
    """Returns row partitions representing this shape.

    In order to represent a shape as row partitions, the rank of the shape
    must be known, and the shape must have rank at least one.

    Returns:
      A list of RowPartition objects.
    Raises:
      ValueError, if the shape cannot be represented by RowPartitions.
    """
    rank = self.rank
    if rank is None:
        raise ValueError('rank must be known for _as_row_partitions')
    elif rank < 1:
        raise ValueError('rank must be >= 1 for _as_row_partitions')
    fully_ragged = self._with_num_row_partitions(rank - 1)
    return fully_ragged.row_partitions