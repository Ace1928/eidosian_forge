import typing
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_shape
from tensorflow.python.util import dispatch
def _coordinate_where(condition):
    """Ragged version of tf.where(condition)."""
    if not isinstance(condition, ragged_tensor.RaggedTensor):
        return array_ops.where(condition)
    selected_coords = _coordinate_where(condition.values)
    condition = condition.with_row_splits_dtype(selected_coords.dtype)
    first_index = selected_coords[:, 0]
    selected_rows = array_ops.gather(condition.value_rowids(), first_index)
    selected_row_starts = array_ops.gather(condition.row_splits, selected_rows)
    selected_cols = first_index - selected_row_starts
    return array_ops.concat([array_ops.expand_dims(selected_rows, 1), array_ops.expand_dims(selected_cols, 1), selected_coords[:, 1:]], axis=1)