from sympy.concrete.summations import Sum
from sympy.core.symbol import symbols, Symbol, Dummy
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import eye
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from sympy.matrices.expressions.hadamard import HadamardPower
from sympy.matrices.expressions.matexpr import (MatrixSymbol,
from sympy.matrices.expressions.matpow import MatPow
from sympy.matrices.expressions.special import (ZeroMatrix, Identity,
from sympy.matrices.expressions.trace import Trace, trace
from sympy.matrices.immutable import ImmutableMatrix
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
from sympy.testing.pytest import XFAIL, raises
def test_pow_index():
    Q = MatPow(A, 2)
    assert Q[0, 0] == A[0, 0] ** 2 + A[0, 1] * A[1, 0]
    n = symbols('n')
    Q2 = A ** n
    assert Q2[0, 0] == 2 * (-sqrt((A[0, 0] + A[1, 1]) ** 2 - 4 * A[0, 0] * A[1, 1] + 4 * A[0, 1] * A[1, 0]) / 2 + A[0, 0] / 2 + A[1, 1] / 2) ** n * A[0, 1] * A[1, 0] / ((sqrt(A[0, 0] ** 2 - 2 * A[0, 0] * A[1, 1] + 4 * A[0, 1] * A[1, 0] + A[1, 1] ** 2) + A[0, 0] - A[1, 1]) * sqrt(A[0, 0] ** 2 - 2 * A[0, 0] * A[1, 1] + 4 * A[0, 1] * A[1, 0] + A[1, 1] ** 2)) - 2 * (sqrt((A[0, 0] + A[1, 1]) ** 2 - 4 * A[0, 0] * A[1, 1] + 4 * A[0, 1] * A[1, 0]) / 2 + A[0, 0] / 2 + A[1, 1] / 2) ** n * A[0, 1] * A[1, 0] / ((-sqrt(A[0, 0] ** 2 - 2 * A[0, 0] * A[1, 1] + 4 * A[0, 1] * A[1, 0] + A[1, 1] ** 2) + A[0, 0] - A[1, 1]) * sqrt(A[0, 0] ** 2 - 2 * A[0, 0] * A[1, 1] + 4 * A[0, 1] * A[1, 0] + A[1, 1] ** 2))