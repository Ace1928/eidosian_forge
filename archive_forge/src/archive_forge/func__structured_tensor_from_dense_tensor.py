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
def _structured_tensor_from_dense_tensor(t):
    """Create a structured tensor with the shape of a dense tensor."""
    if t.shape.is_fully_defined():
        return StructuredTensor.from_fields({}, shape=t.shape)
    elif t.shape.rank is None:
        raise ValueError("Can't build StructuredTensor w/ unknown rank")
    elif t.shape.rank == 1:
        return StructuredTensor.from_fields({}, shape=t.shape, nrows=array_ops.shape(t)[0])
    else:
        rt = ragged_tensor.RaggedTensor.from_tensor(t)
        return _structured_tensor_from_row_partitions(t.shape, rt._nested_row_partitions)