from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_config
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
def _ragged_tile_axis(rt_input, axis, repeats, row_splits_dtype):
    """Tile a dimension of a RaggedTensor to match a ragged shape."""
    assert axis > 0
    if not ragged_tensor.is_ragged(rt_input):
        rt_input = ragged_tensor.RaggedTensor.from_tensor(rt_input, ragged_rank=1, row_splits_dtype=row_splits_dtype)
    if axis > 1:
        return rt_input.with_values(_ragged_tile_axis(rt_input.values, axis - 1, repeats, row_splits_dtype))
    else:
        src_row_splits = rt_input.nested_row_splits
        src_row_lengths = rt_input.nested_row_lengths()
        splits = src_row_splits[0]
        dst_row_lengths = [repeats]
        for i in range(1, len(src_row_lengths)):
            dst_row_lengths.append(ragged_util.repeat_ranges(src_row_lengths[i], splits, repeats))
            splits = array_ops.gather(src_row_splits[i], splits)
        dst_values = ragged_util.repeat_ranges(rt_input.flat_values, splits, repeats)
        return ragged_tensor.RaggedTensor.from_nested_row_lengths(dst_values, dst_row_lengths, validate=False)