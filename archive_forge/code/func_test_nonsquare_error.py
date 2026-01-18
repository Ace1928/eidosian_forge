from sympy.core import symbols, S
from sympy.matrices.expressions import MatrixSymbol, Inverse, MatPow, ZeroMatrix, OneMatrix
from sympy.matrices.common import NonInvertibleMatrixError, NonSquareMatrixError
from sympy.matrices import eye, Identity
from sympy.testing.pytest import raises
from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine
def test_nonsquare_error():
    A = MatrixSymbol('A', 3, 4)
    raises(NonSquareMatrixError, lambda: Inverse(A))