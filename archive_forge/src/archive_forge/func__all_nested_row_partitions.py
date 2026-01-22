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
def _all_nested_row_partitions(rt):
    """Returns all nested row partitions in rt, including for dense dimensions."""
    if isinstance(rt, tensor_lib.Tensor):
        if rt.shape.rank <= 1:
            return ()
        else:
            rt2 = ragged_tensor.RaggedTensor.from_tensor(rt)
            return rt2._nested_row_partitions
    else:
        tail_partitions = _all_nested_row_partitions(rt.flat_values)
        head_partitions = rt._nested_row_partitions
        return head_partitions + tail_partitions