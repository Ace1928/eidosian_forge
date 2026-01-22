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
def _add_row_partitions(self, flat_values, validate=False):
    """Add row partitions to flat_values, if necessary.

    If the shape is truly ragged, then this adds the row_partitions.

    The shape is dense, then this just returns flat_values.

    Args:
      flat_values: the flat_values of a ragged tensor with this shape, or a
        dense tensor with this shape.
      validate: validate the flat_values have the right first dimension.

    Returns:
      flat_values reshaped to have row_partitions.
    """
    if self.row_partitions:
        if validate:
            flat_values = self._validate_flat_values(flat_values)
        return ragged_tensor.RaggedTensor._from_nested_row_partitions(flat_values, self.row_partitions, validate=False)
    else:
        return flat_values