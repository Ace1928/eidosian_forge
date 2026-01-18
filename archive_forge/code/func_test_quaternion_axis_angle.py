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
def test_quaternion_axis_angle():
    test_data = [((1, 0, 0), 0, (1, 0, 0, 0)), ((1, 0, 0), pi / 2, (sqrt(2) / 2, sqrt(2) / 2, 0, 0)), ((0, 1, 0), pi / 2, (sqrt(2) / 2, 0, sqrt(2) / 2, 0)), ((0, 0, 1), pi / 2, (sqrt(2) / 2, 0, 0, sqrt(2) / 2)), ((1, 0, 0), pi, (0, 1, 0, 0)), ((0, 1, 0), pi, (0, 0, 1, 0)), ((0, 0, 1), pi, (0, 0, 0, 1)), ((1, 1, 1), pi, (0, 1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3))), ((sqrt(3) / 3, sqrt(3) / 3, sqrt(3) / 3), pi * 2 / 3, (S.Half, S.Half, S.Half, S.Half))]
    for axis, angle, expected in test_data:
        assert Quaternion.from_axis_angle(axis, angle) == Quaternion(*expected)