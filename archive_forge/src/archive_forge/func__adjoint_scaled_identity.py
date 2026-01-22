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
@linear_operator_algebra.RegisterAdjoint(linear_operator_identity.LinearOperatorScaledIdentity)
def _adjoint_scaled_identity(identity_operator):
    multiplier = identity_operator.multiplier
    if multiplier.dtype.is_complex:
        multiplier = math_ops.conj(multiplier)
    return linear_operator_identity.LinearOperatorScaledIdentity(num_rows=identity_operator._num_rows, multiplier=multiplier, is_non_singular=identity_operator.is_non_singular, is_self_adjoint=identity_operator.is_self_adjoint, is_positive_definite=identity_operator.is_positive_definite, is_square=True)