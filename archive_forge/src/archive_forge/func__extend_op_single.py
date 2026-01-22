from typing import Sequence
from tensorflow.core.config import flags
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.structured.structured_tensor import StructuredTensor
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
def _extend_op_single(value, leaf_op, empty_st_op=None):
    """Extend an op to a value instead of a list of values."""

    def to_list_op(element_op):
        if element_op is None:
            return None

        def list_op(values):
            [value] = values
            return element_op(value)
        return list_op
    return _extend_op([value], to_list_op(leaf_op), to_list_op(empty_st_op))