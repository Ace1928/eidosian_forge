from sympy.functions.elementary.miscellaneous import sqrt
from sympy.simplify.powsimp import powsimp
from sympy.testing.pytest import raises
from sympy.core.expr import unchanged
from sympy.core import symbols, S
from sympy.matrices import Identity, MatrixSymbol, ImmutableMatrix, ZeroMatrix, OneMatrix, Matrix
from sympy.matrices.common import NonSquareMatrixError
from sympy.matrices.expressions import MatPow, MatAdd, MatMul
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matexpr import MatrixElement
def test_nonsquare():
    A = MatrixSymbol('A', 2, 3)
    B = ImmutableMatrix([[1, 2, 3], [4, 5, 6]])
    for r in [-1, 0, 1, 2, S.Half, S.Pi, n]:
        raises(NonSquareMatrixError, lambda: MatPow(A, r))
        raises(NonSquareMatrixError, lambda: MatPow(B, r))