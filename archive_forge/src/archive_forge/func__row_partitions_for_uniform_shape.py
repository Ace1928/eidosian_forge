import re
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _row_partitions_for_uniform_shape(shape, rank):
    """Returns row partitions for the given shape Tensor.

  Args:
    shape: A vector describing a uniform shape.
    rank: The number of dimensions to generate row partitions for

  Returns:
    A list of (rank-1) `RowPartition`s with uniform row length.
  """
    shape_cumprod = math_ops.cumprod(shape[:rank])
    return tuple([RowPartition.from_uniform_row_length(uniform_row_length=shape[i + 1], nvals=shape_cumprod[i + 1], nrows=shape_cumprod[i]) for i in range(rank - 1)])