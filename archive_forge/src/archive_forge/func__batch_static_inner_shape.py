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
def _batch_static_inner_shape(old_shape: tensor_shape.TensorShape, batch_size: Optional[int]) -> tensor_shape.TensorShape:
    """Returns a copy of old_shape with axis=0 multiplied by batch_size.

  Only use if this is the inner_shape of a DynamicRaggedShape.Spec with one
  or more row partitions.

  Args:
    old_shape: the original inner_shape.
    batch_size: the batch size.

  Returns:
    a new shape.
  """
    head_dim = tensor_shape.dimension_at_index(old_shape, 0) * batch_size
    return head_dim + old_shape[1:]