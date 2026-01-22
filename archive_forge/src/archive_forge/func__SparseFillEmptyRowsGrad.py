from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
@ops.RegisterGradient('SparseFillEmptyRows')
def _SparseFillEmptyRowsGrad(op, unused_grad_output_indices, output_grad_values, unused_grad_empty_row_indicator, unused_grad_reverse_index_map):
    """Gradients for SparseFillEmptyRows."""
    reverse_index_map = op.outputs[3]
    d_values, d_default_value = gen_sparse_ops.sparse_fill_empty_rows_grad(reverse_index_map=reverse_index_map, grad_values=output_grad_values)
    return [None, d_values, None, d_default_value]