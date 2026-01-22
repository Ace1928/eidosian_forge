from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_block_diag
from tensorflow.python.ops.linalg import linear_operator_circulant
from tensorflow.python.ops.linalg import linear_operator_composition
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_inversion
from tensorflow.python.ops.linalg import linear_operator_lower_triangular
from tensorflow.python.ops.linalg import registrations_util
@linear_operator_algebra.RegisterSolve(linear_operator_inversion.LinearOperatorInversion, linear_operator.LinearOperator)
def _solve_inverse_linear_operator(linop_a, linop_b):
    """Solve inverse of generic `LinearOperator`s."""
    return linop_a.operator.matmul(linop_b)