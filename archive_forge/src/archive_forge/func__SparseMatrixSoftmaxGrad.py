from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
@ops.RegisterGradient('SparseMatrixSoftmax')
def _SparseMatrixSoftmaxGrad(op, grad_softmax):
    """Gradient for sparse_matrix_softmax op."""
    softmax = op.outputs[0]
    return sparse_csr_matrix_ops.sparse_matrix_softmax_grad(softmax, grad_softmax, type=op.get_attr('type'))