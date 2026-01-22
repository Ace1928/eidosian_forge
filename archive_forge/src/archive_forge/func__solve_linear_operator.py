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
@linear_operator_algebra.RegisterSolve(linear_operator.LinearOperator, linear_operator.LinearOperator)
def _solve_linear_operator(linop_a, linop_b):
    """Generic solve of two `LinearOperator`s."""
    is_square = registrations_util.is_square(linop_a, linop_b)
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    if is_square:
        is_non_singular = registrations_util.combined_non_singular_hint(linop_a, linop_b)
    elif is_square is False:
        is_non_singular = False
        is_self_adjoint = False
        is_positive_definite = False
    return linear_operator_composition.LinearOperatorComposition(operators=[linear_operator_inversion.LinearOperatorInversion(linop_a), linop_b], is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=is_square)