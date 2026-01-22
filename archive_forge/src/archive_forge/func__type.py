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
def _type(operator):
    """Returns the type name constant (e.g. _TRIL) for operator."""
    if isinstance(operator, linear_operator_diag.LinearOperatorDiag):
        return _DIAG
    if isinstance(operator, linear_operator_lower_triangular.LinearOperatorLowerTriangular):
        return _TRIL
    if isinstance(operator, linear_operator_full_matrix.LinearOperatorFullMatrix):
        return _MATRIX
    if isinstance(operator, linear_operator_identity.LinearOperatorIdentity):
        return _IDENTITY
    if isinstance(operator, linear_operator_identity.LinearOperatorScaledIdentity):
        return _SCALED_IDENTITY
    raise TypeError(f'Expected operator to be one of [LinearOperatorDiag, LinearOperatorLowerTriangular, LinearOperatorFullMatrix, LinearOperatorIdentity, LinearOperatorScaledIdentity]. Received: {operator}')