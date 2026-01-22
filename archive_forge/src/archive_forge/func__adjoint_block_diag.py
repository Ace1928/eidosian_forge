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
@linear_operator_algebra.RegisterAdjoint(linear_operator_block_diag.LinearOperatorBlockDiag)
def _adjoint_block_diag(block_diag_operator):
    return linear_operator_block_diag.LinearOperatorBlockDiag(operators=[operator.adjoint() for operator in block_diag_operator.operators], is_non_singular=block_diag_operator.is_non_singular, is_self_adjoint=block_diag_operator.is_self_adjoint, is_positive_definite=block_diag_operator.is_positive_definite, is_square=True)