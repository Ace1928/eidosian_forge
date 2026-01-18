from sympy.core.evalf import N
from sympy.core.numbers import (Float, I, Rational)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices import eye, Matrix
from sympy.core.singleton import S
from sympy.testing.pytest import raises, XFAIL
from sympy.matrices.matrices import NonSquareMatrixError, MatrixError
from sympy.matrices.expressions.fourier import DFT
from sympy.simplify.simplify import simplify
from sympy.matrices.immutable import ImmutableMatrix
from sympy.testing.pytest import slow
from sympy.testing.matrices import allclose
def test_positive_semidefinite_cholesky():
    from sympy.matrices.eigen import _is_positive_semidefinite_cholesky
    m = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert _is_positive_semidefinite_cholesky(m) == True
    m = Matrix([[0, 0, 0], [0, 5, -10 * I], [0, 10 * I, 5]])
    assert _is_positive_semidefinite_cholesky(m) == False
    m = Matrix([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    assert _is_positive_semidefinite_cholesky(m) == False
    m = Matrix([[0, 1], [1, 0]])
    assert _is_positive_semidefinite_cholesky(m) == False
    m = Matrix([[4, -2, -6], [-2, 10, 9], [-6, 9, 14]])
    assert _is_positive_semidefinite_cholesky(m) == True
    m = Matrix([[9, -3, 3], [-3, 2, 1], [3, 1, 6]])
    assert _is_positive_semidefinite_cholesky(m) == True
    m = Matrix([[4, -2, 2], [-2, 1, -1], [2, -1, 5]])
    assert _is_positive_semidefinite_cholesky(m) == True
    m = Matrix([[1, 2, -1], [2, 5, 1], [-1, 1, 9]])
    assert _is_positive_semidefinite_cholesky(m) == False