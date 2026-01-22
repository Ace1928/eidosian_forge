from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
@ops.RegisterGradient('SparseMatrixSparseMatMul')
def _SparseMatrixSparseMatMulGrad(op, grad):
    """Gradient for sparse_matrix_sparse_mat_mul op."""
    t_a = op.get_attr('transpose_a')
    t_b = op.get_attr('transpose_b')
    adj_a = op.get_attr('adjoint_a')
    adj_b = op.get_attr('adjoint_b')
    dtype = op.get_attr('type')
    a = op.inputs[0]
    b = op.inputs[1]
    conj = math_ops.conj
    matmul = sparse_csr_matrix_ops.sparse_matrix_sparse_mat_mul
    if not t_a and (not t_b):
        if not adj_a:
            if not adj_b:
                grad_a = matmul(grad, b, adjoint_b=True, type=dtype)
                grad_b = matmul(a, grad, adjoint_a=True, type=dtype)
            else:
                grad_a = matmul(grad, b, type=dtype)
                grad_b = matmul(grad, a, adjoint_a=True, type=dtype)
        elif not adj_b:
            grad_a = matmul(b, grad, adjoint_b=True, type=dtype)
            grad_b = matmul(a, grad, type=dtype)
        else:
            grad_a = matmul(b, grad, adjoint_a=True, adjoint_b=True, type=dtype)
            grad_b = matmul(grad, a, adjoint_a=True, adjoint_b=True, type=dtype)
    elif not adj_a and (not adj_b):
        if not t_a and t_b:
            grad_a = matmul(grad, conj(b), type=dtype)
            grad_b = matmul(grad, conj(a), transpose_a=True, type=dtype)
        elif t_a and (not t_b):
            grad_a = matmul(conj(b), grad, transpose_b=True, type=dtype)
            grad_b = matmul(conj(a), grad, type=dtype)
        else:
            grad_a = matmul(b, grad, adjoint_a=True, transpose_b=True, type=dtype)
            grad_b = matmul(grad, a, transpose_a=True, adjoint_b=True, type=dtype)
    elif adj_a and t_b:
        grad_a = matmul(b, grad, transpose_a=True, adjoint_b=True, type=dtype)
        grad_b = matmul(grad, a, transpose_a=True, transpose_b=True, type=dtype)
    elif t_a and adj_b:
        grad_a = matmul(b, grad, transpose_a=True, transpose_b=True, type=dtype)
        grad_b = matmul(grad, a, adjoint_a=True, transpose_b=True, type=dtype)
    return (_PruneCSRMatrix(grad_a, a), _PruneCSRMatrix(grad_b, b))