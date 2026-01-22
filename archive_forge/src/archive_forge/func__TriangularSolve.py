from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as _linalg
def _TriangularSolve(x, r):
    """Equiv to matmul(x, adjoint(matrix_inverse(r))) if r is upper-tri."""
    return _linalg.adjoint(linalg_ops.matrix_triangular_solve(r, _linalg.adjoint(x), lower=False, adjoint=False))