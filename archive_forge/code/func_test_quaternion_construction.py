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
def test_quaternion_construction():
    q = Quaternion(w, x, y, z)
    assert q + q == Quaternion(2 * w, 2 * x, 2 * y, 2 * z)
    q2 = Quaternion.from_axis_angle((sqrt(3) / 3, sqrt(3) / 3, sqrt(3) / 3), pi * Rational(2, 3))
    assert q2 == Quaternion(S.Half, S.Half, S.Half, S.Half)
    M = Matrix([[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]])
    q3 = trigsimp(Quaternion.from_rotation_matrix(M))
    assert q3 == Quaternion(sqrt(2) * sqrt(cos(phi) + 1) / 2, 0, 0, sqrt(2 - 2 * cos(phi)) * sign(sin(phi)) / 2)
    nc = Symbol('nc', commutative=False)
    raises(ValueError, lambda: Quaternion(w, x, nc, z))