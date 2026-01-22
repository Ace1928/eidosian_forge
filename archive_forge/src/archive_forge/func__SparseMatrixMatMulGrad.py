from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
@ops.RegisterGradient('SparseMatrixMatMul')
def _SparseMatrixMatMulGrad(op, grad):
    """Gradient for sparse_matrix_mat_mul op."""
    t_a = op.get_attr('transpose_a')
    t_b = op.get_attr('transpose_b')
    adj_a = op.get_attr('adjoint_a')
    adj_b = op.get_attr('adjoint_b')
    transpose_output = op.get_attr('transpose_output')
    conjugate_output = op.get_attr('conjugate_output')
    a = op.inputs[0]
    b = op.inputs[1]
    conj = math_ops.conj
    sparse_matmul = sparse_csr_matrix_ops.sparse_matrix_mat_mul

    def matmul(x, y, **kwargs):
        return _PrunedDenseMatrixMultiplication(x, y, indices=sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(a, type=x.dtype).indices, **kwargs)
    if conjugate_output:
        grad = conj(grad)
    if not transpose_output:
        if not adj_a and (not adj_b):
            a = conj(a)
            b = conj(b)
            if not t_a:
                grad_a = matmul(grad, b, transpose_b=not t_b)
            else:
                grad_a = matmul(b, grad, transpose_a=t_b, transpose_b=True)
            grad_b = sparse_matmul(a, grad, transpose_a=not t_a, transpose_output=t_b)
        elif not t_a and (not t_b):
            if not adj_a:
                grad_a = matmul(grad, b, adjoint_b=not adj_b)
            else:
                grad_a = matmul(b, grad, adjoint_a=adj_b, adjoint_b=True)
            grad_b = sparse_matmul(a, grad, adjoint_a=not adj_a, transpose_output=adj_b, conjugate_output=adj_b)
        elif adj_a and t_b:
            grad_a = matmul(b, grad, transpose_a=True, adjoint_b=True)
            grad_b = sparse_matmul(a, grad, transpose_output=True)
        elif t_a and adj_b:
            grad_a = matmul(b, grad, transpose_a=True, transpose_b=True)
            grad_b = sparse_matmul(conj(a), grad, transpose_output=True, conjugate_output=True)
    elif not adj_a and (not adj_b):
        a = conj(a)
        b = conj(b)
        if not t_a:
            grad_a = matmul(grad, b, transpose_a=True, transpose_b=not t_b)
        else:
            grad_a = matmul(b, grad, transpose_a=t_b)
        grad_b = sparse_matmul(a, grad, transpose_a=not t_a, transpose_b=True, transpose_output=t_b)
    elif not t_a and (not t_b):
        if not adj_a:
            grad_a = matmul(grad, b, transpose_a=True, adjoint_b=not adj_b)
        else:
            grad_a = matmul(b, conj(grad), adjoint_a=adj_b)
        grad_b = sparse_matmul(a, grad, adjoint_a=not adj_a, transpose_b=True, transpose_output=adj_b, conjugate_output=adj_b)
    elif adj_a and t_b:
        grad_a = matmul(b, conj(grad), transpose_a=True)
        grad_b = sparse_matmul(a, grad, transpose_b=True, transpose_output=True)
    elif t_a and adj_b:
        grad_a = matmul(b, grad, transpose_a=True)
        grad_b = sparse_matmul(a, grad, adjoint_b=True, transpose_output=True)
    return (grad_a, grad_b)