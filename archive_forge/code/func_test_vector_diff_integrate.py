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
def test_vector_diff_integrate():
    f = Function('f')
    v = f(a) * C.i + a ** 2 * C.j - C.k
    assert Derivative(v, a) == Derivative(f(a) * C.i + a ** 2 * C.j + -1 * C.k, a)
    assert diff(v, a) == v.diff(a) == Derivative(v, a).doit() == Derivative(f(a), a) * C.i + 2 * a * C.j
    assert Integral(v, a) == Integral(f(a), a) * C.i + Integral(a ** 2, a) * C.j + Integral(-1, a) * C.k