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
def _ragged_stack_concat_axis_0(rt_inputs, stack_values):
    """Helper function to concatenate or stack ragged tensors along axis 0.

  Args:
    rt_inputs: A list of RaggedTensors, all with the same rank and ragged_rank.
    stack_values: Boolean.  If true, then stack values; otherwise, concatenate
      them.

  Returns:
    A RaggedTensor.
  """
    flat_values = [rt.flat_values for rt in rt_inputs]
    concatenated_flat_values = array_ops.concat(flat_values, axis=0)
    nested_splits = [rt.nested_row_splits for rt in rt_inputs]
    ragged_rank = rt_inputs[0].ragged_rank
    concatenated_nested_splits = [_concat_ragged_splits([ns[dim] for ns in nested_splits]) for dim in range(ragged_rank)]
    if stack_values:
        stack_lengths = array_ops_stack.stack([rt.nrows() for rt in rt_inputs])
        stack_splits = ragged_util.lengths_to_splits(stack_lengths)
        concatenated_nested_splits.insert(0, stack_splits)
    return ragged_tensor.RaggedTensor.from_nested_row_splits(concatenated_flat_values, concatenated_nested_splits, validate=False)