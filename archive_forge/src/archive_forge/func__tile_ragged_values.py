from typing import Optional
from typing import Union
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_ragged_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _tile_ragged_values(rt_input, multiples, const_multiples=None):
    """Builds flat_values tensor for a tiled `RaggedTensor`.

  Returns a tensor that repeats the values in
  `rt_input.flat_values` in the
  appropriate pattern to construct a `RaggedTensor` that tiles `rt_input` as
  specified by `multiples`.

  Args:
    rt_input: The `RaggedTensor` whose values should be repeated.
    multiples: A 1-D integer `tensor`, indicating how many times each dimension
      should be repeated.
    const_multiples: Optional constant value for multiples.  Used to skip tiling
      dimensions where `multiples=1`.

  Returns:
    A `Tensor` with the same type and rank as `rt_input.flat_values`.

  #### Example:

  >>> rt = tf.ragged.constant([[1, 2], [3]])
  >>> _tile_ragged_values(rt, tf.constant([3, 2])).numpy()
  array([1, 2, 1, 2, 3, 3, 1, 2, 1, 2, 3, 3, 1, 2, 1, 2, 3, 3], dtype=int32)
  """
    ragged_rank = rt_input.ragged_rank
    nested_splits = rt_input.nested_row_splits
    inner_value_ids = math_ops.range(nested_splits[-1][-1])
    prev_splits = None
    for axis in range(ragged_rank, 0, -1):
        splits = nested_splits[axis - 1]
        if prev_splits is not None:
            splits = array_ops.gather(prev_splits * multiples[axis + 1], splits)
        if const_multiples is None or const_multiples[axis] != 1:
            inner_value_ids = ragged_util.repeat_ranges(inner_value_ids, splits, multiples[axis])
        prev_splits = splits
    ragged_tiled_values = array_ops.gather(rt_input.flat_values, inner_value_ids)
    inner_repeats = array_ops.concat([multiples[:1], multiples[ragged_rank + 1:]], axis=0)
    return array_ops.tile(ragged_tiled_values, inner_repeats)