from sympy.functions import adjoint, conjugate, transpose
from sympy.matrices.expressions import MatrixSymbol, Adjoint, trace, Transpose
from sympy.matrices import eye, Matrix
from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine
from sympy.core.singleton import S
from sympy.core.symbol import symbols
def test_transpose_MatAdd_MatMul():
    from sympy.functions.elementary.trigonometric import cos
    x = symbols('x')
    M = MatrixSymbol('M', 3, 3)
    N = MatrixSymbol('N', 3, 3)
    assert (N + cos(x) * M).T == cos(x) * M.T + N.T