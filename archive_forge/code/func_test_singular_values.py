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
def test_singular_values():
    x = Symbol('x', real=True)
    A = Matrix([[0, 1 * I], [2, 0]])
    assert A.singular_values() == [2, 1]
    A = eye(3)
    A[1, 1] = x
    A[2, 2] = 5
    vals = A.singular_values()
    assert set(vals) == {5, 1, Abs(x)}
    A = Matrix([[sin(x), cos(x)], [-cos(x), sin(x)]])
    vals = [sv.trigsimp() for sv in A.singular_values()]
    assert vals == [S.One, S.One]
    A = Matrix([[2, 4], [1, 3], [0, 0], [0, 0]])
    assert A.singular_values() == [sqrt(sqrt(221) + 15), sqrt(15 - sqrt(221))]
    assert A.T.singular_values() == [sqrt(sqrt(221) + 15), sqrt(15 - sqrt(221)), 0, 0]