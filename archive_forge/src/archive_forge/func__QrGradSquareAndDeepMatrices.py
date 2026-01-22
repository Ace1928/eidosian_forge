from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as _linalg
def _QrGradSquareAndDeepMatrices(q, r, dq, dr):
    """Gradient for matrix orders num_rows >= num_cols
    and full_matrices is false.
    """
    qdq = math_ops.matmul(q, dq, adjoint_a=True)
    qdq_ = qdq - _linalg.adjoint(qdq)
    rdr = math_ops.matmul(r, dr, adjoint_b=True)
    rdr_ = rdr - _linalg.adjoint(rdr)
    tril = array_ops.matrix_band_part(qdq_ + rdr_, -1, 0)
    grad_a = math_ops.matmul(q, dr + _TriangularSolve(tril, r))
    grad_b = _TriangularSolve(dq - math_ops.matmul(q, qdq), r)
    ret = grad_a + grad_b
    if q.dtype.is_complex:
        m = rdr - _linalg.adjoint(qdq)
        eyem = _linalg.set_diag(array_ops.zeros_like(m), _linalg.diag_part(m))
        correction = eyem - math_ops.cast(math_ops.real(eyem), q.dtype)
        ret = ret + _TriangularSolve(math_ops.matmul(q, _linalg.adjoint(correction)), r)
    return ret