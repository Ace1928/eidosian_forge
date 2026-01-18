from sympy.concrete.summations import Sum
from sympy.core.exprtools import gcd_terms
from sympy.core.function import (diff, expand)
from sympy.core.relational import Eq
from sympy.core.symbol import (Dummy, Symbol, Str)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import zeros
from sympy.polys.polytools import factor
from sympy.core import (S, symbols, Add, Mul, SympifyError, Rational,
from sympy.functions import sin, cos, tan, sqrt, cbrt, exp
from sympy.simplify import simplify
from sympy.matrices import (ImmutableMatrix, Inverse, MatAdd, MatMul,
from sympy.matrices.common import NonSquareMatrixError
from sympy.matrices.expressions.determinant import Determinant, det
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.matrices.expressions.special import ZeroMatrix, Identity
from sympy.testing.pytest import raises, XFAIL
def test_single_indexing():
    A = MatrixSymbol('A', 2, 3)
    assert A[1] == A[0, 1]
    assert A[int(1)] == A[0, 1]
    assert A[3] == A[1, 0]
    assert list(A[:2, :2]) == [A[0, 0], A[0, 1], A[1, 0], A[1, 1]]
    raises(IndexError, lambda: A[6])
    raises(IndexError, lambda: A[n])
    B = MatrixSymbol('B', n, m)
    raises(IndexError, lambda: B[1])
    B = MatrixSymbol('B', n, 3)
    assert B[3] == B[1, 0]