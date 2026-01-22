from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_addition
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_block_diag
from tensorflow.python.ops.linalg import linear_operator_block_lower_triangular
from tensorflow.python.ops.linalg import linear_operator_circulant
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_full_matrix
from tensorflow.python.ops.linalg import linear_operator_householder
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_inversion
from tensorflow.python.ops.linalg import linear_operator_kronecker
@linear_operator_algebra.RegisterInverse(linear_operator_identity.LinearOperatorScaledIdentity)
def _inverse_scaled_identity(identity_operator):
    return linear_operator_identity.LinearOperatorScaledIdentity(num_rows=identity_operator._num_rows, multiplier=1.0 / identity_operator.multiplier, is_non_singular=identity_operator.is_non_singular, is_self_adjoint=True, is_positive_definite=identity_operator.is_positive_definite, is_square=True)