from sympy.core.function import diff
from sympy.core.function import expand
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (Abs, conjugate, im, re, sign)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, cos, sin, atan2, atan)
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import Matrix
from sympy.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.algebras.quaternion import Quaternion
from sympy.testing.pytest import raises
from itertools import permutations, product
def test_quaternion_axis_angle_simplification():
    result = Quaternion.from_axis_angle((1, 2, 3), asin(4))
    assert result.a == cos(asin(4) / 2)
    assert result.b == sqrt(14) * sin(asin(4) / 2) / 14
    assert result.c == sqrt(14) * sin(asin(4) / 2) / 7
    assert result.d == 3 * sqrt(14) * sin(asin(4) / 2) / 14