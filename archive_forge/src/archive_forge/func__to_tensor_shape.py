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
def _to_tensor_shape(self):
    """Get a tensor shape corresponding to this type."""
    alt = self
    if alt._static_inner_shape.rank is None:
        return tensor_shape.TensorShape(None)
    if alt._static_inner_shape.rank == 0:
        assert not alt._row_partitions
        return alt._static_inner_shape
    prefix = [alt._dimension(0)]
    prefix.extend([rp.uniform_row_length for rp in alt._row_partitions])
    suffix = alt._static_inner_shape[1:]
    return tensor_shape.TensorShape(prefix) + suffix