from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_adjoint
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_block_diag
from tensorflow.python.ops.linalg import linear_operator_circulant
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_householder
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_kronecker
@linear_operator_algebra.RegisterAdjoint(linear_operator_diag.LinearOperatorDiag)
def _adjoint_diag(diag_operator):
    diag = diag_operator.diag
    if diag.dtype.is_complex:
        diag = math_ops.conj(diag)
    return linear_operator_diag.LinearOperatorDiag(diag=diag, is_non_singular=diag_operator.is_non_singular, is_self_adjoint=diag_operator.is_self_adjoint, is_positive_definite=diag_operator.is_positive_definite, is_square=True)