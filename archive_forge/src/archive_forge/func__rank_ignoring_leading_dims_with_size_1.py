from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
def _rank_ignoring_leading_dims_with_size_1(value):
    """Returns `rank(value)`, ignoring any leading dimensions with size 1."""
    if value.shape.rank is not None:
        ndims = value.shape.rank
        for dim in value.shape.dims:
            if dim.value == 1:
                ndims -= 1
            elif dim.value is None:
                ndims = None
                break
            else:
                break
        if ndims is not None:
            return ndims
    shape = array_ops.shape(value)
    dim_is_one = math_ops.cast(math_ops.equal(shape, 1), dtypes.int32)
    leading_ones = math_ops.cumprod(dim_is_one)
    num_leading_ones = math_ops.reduce_sum(leading_ones)
    return array_ops.rank(value) - num_leading_ones