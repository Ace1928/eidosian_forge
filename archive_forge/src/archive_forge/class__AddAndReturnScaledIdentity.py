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
class _AddAndReturnScaledIdentity(_Adder):
    """Handles additions resulting in an Identity family member.

  The Identity (`LinearOperatorScaledIdentity`, `LinearOperatorIdentity`) family
  is closed under addition.  This `Adder` respects that, and returns an Identity
  """

    def can_add(self, op1, op2):
        types = {_type(op1), _type(op2)}
        return not types.difference(_IDENTITY_FAMILY)

    def _add(self, op1, op2, operator_name, hints):
        if _type(op1) == _SCALED_IDENTITY:
            multiplier_1 = op1.multiplier
        else:
            multiplier_1 = array_ops.ones(op1.batch_shape_tensor(), dtype=op1.dtype)
        if _type(op2) == _SCALED_IDENTITY:
            multiplier_2 = op2.multiplier
        else:
            multiplier_2 = array_ops.ones(op2.batch_shape_tensor(), dtype=op2.dtype)
        return linear_operator_identity.LinearOperatorScaledIdentity(num_rows=op1.range_dimension_tensor(), multiplier=multiplier_1 + multiplier_2, is_non_singular=hints.is_non_singular, is_self_adjoint=hints.is_self_adjoint, is_positive_definite=hints.is_positive_definite, name=operator_name)