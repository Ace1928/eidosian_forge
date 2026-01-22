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
@linear_operator_algebra.RegisterAdjoint(linear_operator_circulant._BaseLinearOperatorCirculant)
def _adjoint_circulant(circulant_operator):
    spectrum = circulant_operator.spectrum
    if spectrum.dtype.is_complex:
        spectrum = math_ops.conj(spectrum)
    return circulant_operator.__class__(spectrum=spectrum, is_non_singular=circulant_operator.is_non_singular, is_self_adjoint=circulant_operator.is_self_adjoint, is_positive_definite=circulant_operator.is_positive_definite, is_square=True)