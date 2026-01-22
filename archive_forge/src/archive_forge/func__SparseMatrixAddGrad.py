from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
@ops.RegisterGradient('SparseMatrixAdd')
def _SparseMatrixAddGrad(op, grad):
    """Gradient for sparse_matrix_add op."""
    a_csr, b_csr, alpha, beta = op.inputs
    return (sparse_csr_matrix_ops.sparse_matrix_mul(_PruneCSRMatrix(grad, a_csr), alpha), sparse_csr_matrix_ops.sparse_matrix_mul(_PruneCSRMatrix(grad, b_csr), beta), None, None)