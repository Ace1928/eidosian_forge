from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
@ops.RegisterGradient('SparseReorder')
def _SparseReorderGrad(op, unused_output_indices_grad, output_values_grad):
    """Gradients for the SparseReorder op.

  Args:
    op: the SparseReorder op
    unused_output_indices_grad: the incoming gradients of the output indices
    output_values_grad: the incoming gradients of the output values

  Returns:
    Gradient for each of the 3 input tensors:
      (input_indices, input_values, input_shape)
    The gradients for input_indices and input_shape is None.
  """
    input_indices = op.inputs[0]
    input_shape = op.inputs[2]
    num_entries = array_ops.shape(input_indices)[0]
    entry_indices = math_ops.range(num_entries)
    sp_unordered = sparse_tensor.SparseTensor(input_indices, entry_indices, input_shape)
    sp_ordered = sparse_ops.sparse_reorder(sp_unordered)
    inverted_permutation = array_ops.invert_permutation(sp_ordered.values)
    return (None, array_ops.gather(output_values_grad, inverted_permutation), None)