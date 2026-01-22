import abc
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_full_matrix
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_lower_triangular
def _infer_hints_allowing_override(op1, op2, hints):
    """Infer hints from op1 and op2.  hints argument is an override.

  Args:
    op1:  LinearOperator
    op2:  LinearOperator
    hints:  _Hints object holding "is_X" boolean hints to use for returned
      operator.
      If some hint is None, try to set using op1 and op2.  If the
      hint is provided, ignore op1 and op2 hints.  This allows an override
      of previous hints, but does not allow forbidden hints (e.g. you still
      cannot say a real diagonal operator is not self-adjoint.

  Returns:
    _Hints object.
  """
    hints = hints or _Hints()
    if hints.is_self_adjoint is None:
        is_self_adjoint = op1.is_self_adjoint and op2.is_self_adjoint
    else:
        is_self_adjoint = hints.is_self_adjoint
    if hints.is_positive_definite is None:
        is_positive_definite = op1.is_positive_definite and op2.is_positive_definite
    else:
        is_positive_definite = hints.is_positive_definite
    if is_positive_definite and hints.is_positive_definite is None:
        is_non_singular = True
    else:
        is_non_singular = hints.is_non_singular
    return _Hints(is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite)