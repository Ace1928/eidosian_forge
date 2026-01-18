from sympy.core import Rational, S
from sympy.simplify import simplify, trigsimp
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.vector.vector import Vector, BaseVector, VectorAdd, \
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.vector import Cross, Dot, cross
from sympy.testing.pytest import raises
def test_vector_simplify():
    A, s, k, m = symbols('A, s, k, m')
    test1 = (1 / a + 1 / b) * i
    assert test1 & i != (a + b) / (a * b)
    test1 = simplify(test1)
    assert test1 & i == (a + b) / (a * b)
    assert test1.simplify() == simplify(test1)
    test2 = A ** 2 * s ** 4 / (4 * pi * k * m ** 3) * i
    test2 = simplify(test2)
    assert test2 & i == A ** 2 * s ** 4 / (4 * pi * k * m ** 3)
    test3 = (4 + 4 * a - 2 * (2 + 2 * a)) / (2 + 2 * a) * i
    test3 = simplify(test3)
    assert test3 & i == 0
    test4 = (-4 * a * b ** 2 - 2 * b ** 3 - 2 * a ** 2 * b) / (a + b) ** 2 * i
    test4 = simplify(test4)
    assert test4 & i == -2 * b
    v = (sin(a) + cos(a)) ** 2 * i - j
    assert trigsimp(v) == 2 * sin(a + pi / 4) ** 2 * i + -1 * j
    assert trigsimp(v) == v.trigsimp()
    assert simplify(Vector.zero) == Vector.zero