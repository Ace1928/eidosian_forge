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
def test_quaternion_functions():
    q = Quaternion(w, x, y, z)
    q1 = Quaternion(1, 2, 3, 4)
    q0 = Quaternion(0, 0, 0, 0)
    assert conjugate(q) == Quaternion(w, -x, -y, -z)
    assert q.norm() == sqrt(w ** 2 + x ** 2 + y ** 2 + z ** 2)
    assert q.normalize() == Quaternion(w, x, y, z) / sqrt(w ** 2 + x ** 2 + y ** 2 + z ** 2)
    assert q.inverse() == Quaternion(w, -x, -y, -z) / (w ** 2 + x ** 2 + y ** 2 + z ** 2)
    assert q.inverse() == q.pow(-1)
    raises(ValueError, lambda: q0.inverse())
    assert q.pow(2) == Quaternion(w ** 2 - x ** 2 - y ** 2 - z ** 2, 2 * w * x, 2 * w * y, 2 * w * z)
    assert q ** 2 == Quaternion(w ** 2 - x ** 2 - y ** 2 - z ** 2, 2 * w * x, 2 * w * y, 2 * w * z)
    assert q1.pow(-2) == Quaternion(Rational(-7, 225), Rational(-1, 225), Rational(-1, 150), Rational(-2, 225))
    assert q1 ** (-2) == Quaternion(Rational(-7, 225), Rational(-1, 225), Rational(-1, 150), Rational(-2, 225))
    assert q1.pow(-0.5) == NotImplemented
    raises(TypeError, lambda: q1 ** (-0.5))
    assert q1.exp() == Quaternion(E * cos(sqrt(29)), 2 * sqrt(29) * E * sin(sqrt(29)) / 29, 3 * sqrt(29) * E * sin(sqrt(29)) / 29, 4 * sqrt(29) * E * sin(sqrt(29)) / 29)
    assert q1._ln() == Quaternion(log(sqrt(30)), 2 * sqrt(29) * acos(sqrt(30) / 30) / 29, 3 * sqrt(29) * acos(sqrt(30) / 30) / 29, 4 * sqrt(29) * acos(sqrt(30) / 30) / 29)
    assert q1.pow_cos_sin(2) == Quaternion(30 * cos(2 * acos(sqrt(30) / 30)), 60 * sqrt(29) * sin(2 * acos(sqrt(30) / 30)) / 29, 90 * sqrt(29) * sin(2 * acos(sqrt(30) / 30)) / 29, 120 * sqrt(29) * sin(2 * acos(sqrt(30) / 30)) / 29)
    assert diff(Quaternion(x, x, x, x), x) == Quaternion(1, 1, 1, 1)
    assert integrate(Quaternion(x, x, x, x), x) == Quaternion(x ** 2 / 2, x ** 2 / 2, x ** 2 / 2, x ** 2 / 2)
    assert Quaternion.rotate_point((1, 1, 1), q1) == (S.One / 5, 1, S(7) / 5)
    n = Symbol('n')
    raises(TypeError, lambda: q1 ** n)
    n = Symbol('n', integer=True)
    raises(TypeError, lambda: q1 ** n)
    assert Quaternion(22, 23, 55, 8).scalar_part() == 22
    assert Quaternion(w, x, y, z).scalar_part() == w
    assert Quaternion(22, 23, 55, 8).vector_part() == Quaternion(0, 23, 55, 8)
    assert Quaternion(w, x, y, z).vector_part() == Quaternion(0, x, y, z)
    assert q1.axis() == Quaternion(0, 2 * sqrt(29) / 29, 3 * sqrt(29) / 29, 4 * sqrt(29) / 29)
    assert q1.axis().pow(2) == Quaternion(-1, 0, 0, 0)
    assert q0.axis().scalar_part() == 0
    assert q.axis() == Quaternion(0, x / sqrt(x ** 2 + y ** 2 + z ** 2), y / sqrt(x ** 2 + y ** 2 + z ** 2), z / sqrt(x ** 2 + y ** 2 + z ** 2))
    assert q0.is_pure() is True
    assert q1.is_pure() is False
    assert Quaternion(0, 0, 0, 3).is_pure() is True
    assert Quaternion(0, 2, 10, 3).is_pure() is True
    assert Quaternion(w, 2, 10, 3).is_pure() is None
    assert q1.angle() == atan(sqrt(29))
    assert q.angle() == atan2(sqrt(x ** 2 + y ** 2 + z ** 2), w)
    assert Quaternion.arc_coplanar(q1, Quaternion(2, 4, 6, 8)) is True
    assert Quaternion.arc_coplanar(q1, Quaternion(1, -2, -3, -4)) is True
    assert Quaternion.arc_coplanar(q1, Quaternion(1, 8, 12, 16)) is True
    assert Quaternion.arc_coplanar(q1, Quaternion(1, 2, 3, 4)) is True
    assert Quaternion.arc_coplanar(q1, Quaternion(w, 4, 6, 8)) is True
    assert Quaternion.arc_coplanar(q1, Quaternion(2, 7, 4, 1)) is False
    assert Quaternion.arc_coplanar(q1, Quaternion(w, x, y, z)) is None
    raises(ValueError, lambda: Quaternion.arc_coplanar(q1, q0))
    assert Quaternion.vector_coplanar(Quaternion(0, 8, 12, 16), Quaternion(0, 4, 6, 8), Quaternion(0, 2, 3, 4)) is True
    assert Quaternion.vector_coplanar(Quaternion(0, 0, 0, 0), Quaternion(0, 4, 6, 8), Quaternion(0, 2, 3, 4)) is True
    assert Quaternion.vector_coplanar(Quaternion(0, 8, 2, 6), Quaternion(0, 1, 6, 6), Quaternion(0, 0, 3, 4)) is False
    assert Quaternion.vector_coplanar(Quaternion(0, 1, 3, 4), Quaternion(0, 4, w, 6), Quaternion(0, 6, 8, 1)) is None
    raises(ValueError, lambda: Quaternion.vector_coplanar(q0, Quaternion(0, 4, 6, 8), q1))
    assert Quaternion(0, 1, 2, 3).parallel(Quaternion(0, 2, 4, 6)) is True
    assert Quaternion(0, 1, 2, 3).parallel(Quaternion(0, 2, 2, 6)) is False
    assert Quaternion(0, 1, 2, 3).parallel(Quaternion(w, x, y, 6)) is None
    raises(ValueError, lambda: q0.parallel(q1))
    assert Quaternion(0, 1, 2, 3).orthogonal(Quaternion(0, -2, 1, 0)) is True
    assert Quaternion(0, 2, 4, 7).orthogonal(Quaternion(0, 2, 2, 6)) is False
    assert Quaternion(0, 2, 4, 7).orthogonal(Quaternion(w, x, y, 6)) is None
    raises(ValueError, lambda: q0.orthogonal(q1))
    assert q1.index_vector() == Quaternion(0, 2 * sqrt(870) / 29, 3 * sqrt(870) / 29, 4 * sqrt(870) / 29)
    assert Quaternion(0, 3, 9, 4).index_vector() == Quaternion(0, 3, 9, 4)
    assert Quaternion(4, 3, 9, 4).mensor() == log(sqrt(122))
    assert Quaternion(3, 3, 0, 2).mensor() == log(sqrt(22))
    assert q0.is_zero_quaternion() is True
    assert q1.is_zero_quaternion() is False
    assert Quaternion(w, 0, 0, 0).is_zero_quaternion() is None