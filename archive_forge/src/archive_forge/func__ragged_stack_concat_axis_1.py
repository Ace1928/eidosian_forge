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
def _ragged_stack_concat_axis_1(rt_inputs, stack_values):
    """Helper function to concatenate or stack ragged tensors along axis 1.

  Args:
    rt_inputs: A list of RaggedTensors, all with the same rank and ragged_rank.
    stack_values: Boolean.  If true, then stack values; otherwise, concatenate
      them.

  Returns:
    A RaggedTensor.
  """
    num_inputs = len(rt_inputs)
    nrows_checks = []
    rt_nrows = rt_inputs[0].nrows()
    for index, rt in enumerate(rt_inputs[1:]):
        nrows_checks.append(check_ops.assert_equal(rt_nrows, rt.nrows(), message=f'Input tensors at index 0 (=x) and {index + 1} (=y) have incompatible shapes.'))
    with ops.control_dependencies(nrows_checks):
        concatenated_rt = _ragged_stack_concat_axis_0(rt_inputs, stack_values=False)
        row_indices = math_ops.range(rt_nrows * num_inputs)
        row_index_matrix = array_ops.reshape(row_indices, [num_inputs, -1])
        transposed_row_index_matrix = array_ops.transpose(row_index_matrix)
        row_permutation = array_ops.reshape(transposed_row_index_matrix, [-1])
        permuted_rt = ragged_gather_ops.gather(concatenated_rt, row_permutation)
        if stack_values:
            stack_splits = math_ops.range(0, rt_nrows * num_inputs + 1, num_inputs)
            _copy_row_shape(rt_inputs, stack_splits)
            return ragged_tensor.RaggedTensor.from_row_splits(permuted_rt, stack_splits, validate=False)
        else:
            concat_splits = permuted_rt.row_splits[::num_inputs]
            _copy_row_shape(rt_inputs, concat_splits)
            return ragged_tensor.RaggedTensor.from_row_splits(permuted_rt.values, concat_splits, validate=False)