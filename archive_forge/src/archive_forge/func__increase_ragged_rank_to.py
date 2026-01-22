import typing
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _increase_ragged_rank_to(rt_input, ragged_rank, row_splits_dtype):
    """Adds ragged dimensions to `rt_input` so it has the desired ragged rank."""
    if ragged_rank > 0:
        if not ragged_tensor.is_ragged(rt_input):
            rt_input = ragged_tensor.RaggedTensor.from_tensor(rt_input, row_splits_dtype=row_splits_dtype)
        if rt_input.ragged_rank < ragged_rank:
            rt_input = rt_input.with_values(_increase_ragged_rank_to(rt_input.values, ragged_rank - 1, row_splits_dtype))
    return rt_input