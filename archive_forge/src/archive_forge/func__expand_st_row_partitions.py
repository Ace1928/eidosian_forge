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
def _expand_st_row_partitions(st, axis):
    """Create the row_partitions for expand_dims."""
    if axis == 0:
        if st.shape.rank == 0:
            return ()
        nvals = st.nrows()
        new_partition = RowPartition.from_uniform_row_length(nvals, nvals, nrows=1, validate=False)
        return (new_partition,) + st.row_partitions
    elif axis == st.rank:
        nvals = st.row_partitions[axis - 2].nvals() if axis - 2 >= 0 else st.nrows()
        return st.row_partitions + (RowPartition.from_uniform_row_length(1, nvals, nrows=nvals, validate=False),)
    else:
        nvals = st.row_partitions[axis - 1].nrows() if axis - 1 >= 0 else st.nrows()
        return st.row_partitions[:axis - 1] + (RowPartition.from_uniform_row_length(1, nvals, nrows=nvals, validate=False),) + st.row_partitions[axis - 1:]