from sympy.concrete.summations import Sum
from sympy.core.function import expand
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.abc import alpha, beta, gamma, j, m
from sympy.physics.quantum import hbar, represent, Commutator, InnerProduct
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.spin import (
from sympy.testing.pytest import raises, slow
def test_rotation_small_d():
    assert Rotation.d(S.Half, S.Half, S.Half, beta).doit() == cos(beta / 2)
    assert Rotation.d(S.Half, S.Half, Rational(-1, 2), beta).doit() == -sin(beta / 2)
    assert Rotation.d(S.Half, Rational(-1, 2), S.Half, beta).doit() == sin(beta / 2)
    assert Rotation.d(S.Half, Rational(-1, 2), Rational(-1, 2), beta).doit() == cos(beta / 2)
    assert Rotation.d(1, 1, 1, beta).doit() == (1 + cos(beta)) / 2
    assert Rotation.d(1, 1, 0, beta).doit() == -sin(beta) / sqrt(2)
    assert Rotation.d(1, 1, -1, beta).doit() == (1 - cos(beta)) / 2
    assert Rotation.d(1, 0, 1, beta).doit() == sin(beta) / sqrt(2)
    assert Rotation.d(1, 0, 0, beta).doit() == cos(beta)
    assert Rotation.d(1, 0, -1, beta).doit() == -sin(beta) / sqrt(2)
    assert Rotation.d(1, -1, 1, beta).doit() == (1 - cos(beta)) / 2
    assert Rotation.d(1, -1, 0, beta).doit() == sin(beta) / sqrt(2)
    assert Rotation.d(1, -1, -1, beta).doit() == (1 + cos(beta)) / 2
    assert Rotation.d(S(3) / 2, Rational(3, 2), Rational(3, 2), beta).doit() == (3 * cos(beta / 2) + cos(beta * Rational(3, 2))) / 4
    assert Rotation.d(Rational(3, 2), S(3) / 2, S.Half, beta).doit() == -sqrt(3) * (sin(beta / 2) + sin(beta * Rational(3, 2))) / 4
    assert Rotation.d(Rational(3, 2), S(3) / 2, Rational(-1, 2), beta).doit() == sqrt(3) * (cos(beta / 2) - cos(beta * Rational(3, 2))) / 4
    assert Rotation.d(Rational(3, 2), S(3) / 2, Rational(-3, 2), beta).doit() == (-3 * sin(beta / 2) + sin(beta * Rational(3, 2))) / 4
    assert Rotation.d(Rational(3, 2), S(1) / 2, Rational(3, 2), beta).doit() == sqrt(3) * (sin(beta / 2) + sin(beta * Rational(3, 2))) / 4
    assert Rotation.d(S(3) / 2, S.Half, S.Half, beta).doit() == (cos(beta / 2) + 3 * cos(beta * Rational(3, 2))) / 4
    assert Rotation.d(S(3) / 2, S.Half, Rational(-1, 2), beta).doit() == (sin(beta / 2) - 3 * sin(beta * Rational(3, 2))) / 4
    assert Rotation.d(Rational(3, 2), S(1) / 2, Rational(-3, 2), beta).doit() == sqrt(3) * (cos(beta / 2) - cos(beta * Rational(3, 2))) / 4
    assert Rotation.d(Rational(3, 2), -S(1) / 2, Rational(3, 2), beta).doit() == sqrt(3) * (cos(beta / 2) - cos(beta * Rational(3, 2))) / 4
    assert Rotation.d(Rational(3, 2), -S(1) / 2, S.Half, beta).doit() == (-sin(beta / 2) + 3 * sin(beta * Rational(3, 2))) / 4
    assert Rotation.d(Rational(3, 2), -S(1) / 2, Rational(-1, 2), beta).doit() == (cos(beta / 2) + 3 * cos(beta * Rational(3, 2))) / 4
    assert Rotation.d(Rational(3, 2), -S(1) / 2, Rational(-3, 2), beta).doit() == -sqrt(3) * (sin(beta / 2) + sin(beta * Rational(3, 2))) / 4
    assert Rotation.d(S(3) / 2, Rational(-3, 2), Rational(3, 2), beta).doit() == (3 * sin(beta / 2) - sin(beta * Rational(3, 2))) / 4
    assert Rotation.d(Rational(3, 2), -S(3) / 2, S.Half, beta).doit() == sqrt(3) * (cos(beta / 2) - cos(beta * Rational(3, 2))) / 4
    assert Rotation.d(Rational(3, 2), -S(3) / 2, Rational(-1, 2), beta).doit() == sqrt(3) * (sin(beta / 2) + sin(beta * Rational(3, 2))) / 4
    assert Rotation.d(Rational(3, 2), -S(3) / 2, Rational(-3, 2), beta).doit() == (3 * cos(beta / 2) + cos(beta * Rational(3, 2))) / 4
    assert Rotation.d(2, 2, 2, beta).doit() == (3 + 4 * cos(beta) + cos(2 * beta)) / 8
    assert Rotation.d(2, 2, 1, beta).doit() == -((cos(beta) + 1) * sin(beta)) / 2
    assert Rotation.d(2, 2, 0, beta).doit() == sqrt(6) * sin(beta) ** 2 / 4
    assert Rotation.d(2, 2, -1, beta).doit() == (cos(beta) - 1) * sin(beta) / 2
    assert Rotation.d(2, 2, -2, beta).doit() == (3 - 4 * cos(beta) + cos(2 * beta)) / 8
    assert Rotation.d(2, 1, 2, beta).doit() == (cos(beta) + 1) * sin(beta) / 2
    assert Rotation.d(2, 1, 1, beta).doit() == (cos(beta) + cos(2 * beta)) / 2
    assert Rotation.d(2, 1, 0, beta).doit() == -sqrt(6) * sin(2 * beta) / 4
    assert Rotation.d(2, 1, -1, beta).doit() == (cos(beta) - cos(2 * beta)) / 2
    assert Rotation.d(2, 1, -2, beta).doit() == (cos(beta) - 1) * sin(beta) / 2
    assert Rotation.d(2, 0, 2, beta).doit() == sqrt(6) * sin(beta) ** 2 / 4
    assert Rotation.d(2, 0, 1, beta).doit() == sqrt(6) * sin(2 * beta) / 4
    assert Rotation.d(2, 0, 0, beta).doit() == (1 + 3 * cos(2 * beta)) / 4
    assert Rotation.d(2, 0, -1, beta).doit() == -sqrt(6) * sin(2 * beta) / 4
    assert Rotation.d(2, 0, -2, beta).doit() == sqrt(6) * sin(beta) ** 2 / 4
    assert Rotation.d(2, -1, 2, beta).doit() == (2 * sin(beta) - sin(2 * beta)) / 4
    assert Rotation.d(2, -1, 1, beta).doit() == (cos(beta) - cos(2 * beta)) / 2
    assert Rotation.d(2, -1, 0, beta).doit() == sqrt(6) * sin(2 * beta) / 4
    assert Rotation.d(2, -1, -1, beta).doit() == (cos(beta) + cos(2 * beta)) / 2
    assert Rotation.d(2, -1, -2, beta).doit() == -((cos(beta) + 1) * sin(beta)) / 2
    assert Rotation.d(2, -2, 2, beta).doit() == (3 - 4 * cos(beta) + cos(2 * beta)) / 8
    assert Rotation.d(2, -2, 1, beta).doit() == (2 * sin(beta) - sin(2 * beta)) / 4
    assert Rotation.d(2, -2, 0, beta).doit() == sqrt(6) * sin(beta) ** 2 / 4
    assert Rotation.d(2, -2, -1, beta).doit() == (cos(beta) + 1) * sin(beta) / 2
    assert Rotation.d(2, -2, -2, beta).doit() == (3 + 4 * cos(beta) + cos(2 * beta)) / 8
    assert Rotation.d(S.Half, S.Half, S.Half, pi / 2).doit() == sqrt(2) / 2
    assert Rotation.d(S.Half, S.Half, Rational(-1, 2), pi / 2).doit() == -sqrt(2) / 2
    assert Rotation.d(S.Half, Rational(-1, 2), S.Half, pi / 2).doit() == sqrt(2) / 2
    assert Rotation.d(S.Half, Rational(-1, 2), Rational(-1, 2), pi / 2).doit() == sqrt(2) / 2
    assert Rotation.d(1, 1, 1, pi / 2).doit() == S.Half
    assert Rotation.d(1, 1, 0, pi / 2).doit() == -sqrt(2) / 2
    assert Rotation.d(1, 1, -1, pi / 2).doit() == S.Half
    assert Rotation.d(1, 0, 1, pi / 2).doit() == sqrt(2) / 2
    assert Rotation.d(1, 0, 0, pi / 2).doit() == 0
    assert Rotation.d(1, 0, -1, pi / 2).doit() == -sqrt(2) / 2
    assert Rotation.d(1, -1, 1, pi / 2).doit() == S.Half
    assert Rotation.d(1, -1, 0, pi / 2).doit() == sqrt(2) / 2
    assert Rotation.d(1, -1, -1, pi / 2).doit() == S.Half
    assert Rotation.d(Rational(3, 2), Rational(3, 2), Rational(3, 2), pi / 2).doit() == sqrt(2) / 4
    assert Rotation.d(Rational(3, 2), Rational(3, 2), S.Half, pi / 2).doit() == -sqrt(6) / 4
    assert Rotation.d(Rational(3, 2), Rational(3, 2), Rational(-1, 2), pi / 2).doit() == sqrt(6) / 4
    assert Rotation.d(Rational(3, 2), Rational(3, 2), Rational(-3, 2), pi / 2).doit() == -sqrt(2) / 4
    assert Rotation.d(Rational(3, 2), S.Half, Rational(3, 2), pi / 2).doit() == sqrt(6) / 4
    assert Rotation.d(Rational(3, 2), S.Half, S.Half, pi / 2).doit() == -sqrt(2) / 4
    assert Rotation.d(Rational(3, 2), S.Half, Rational(-1, 2), pi / 2).doit() == -sqrt(2) / 4
    assert Rotation.d(Rational(3, 2), S.Half, Rational(-3, 2), pi / 2).doit() == sqrt(6) / 4
    assert Rotation.d(Rational(3, 2), Rational(-1, 2), Rational(3, 2), pi / 2).doit() == sqrt(6) / 4
    assert Rotation.d(Rational(3, 2), Rational(-1, 2), S.Half, pi / 2).doit() == sqrt(2) / 4
    assert Rotation.d(Rational(3, 2), Rational(-1, 2), Rational(-1, 2), pi / 2).doit() == -sqrt(2) / 4
    assert Rotation.d(Rational(3, 2), Rational(-1, 2), Rational(-3, 2), pi / 2).doit() == -sqrt(6) / 4
    assert Rotation.d(Rational(3, 2), Rational(-3, 2), Rational(3, 2), pi / 2).doit() == sqrt(2) / 4
    assert Rotation.d(Rational(3, 2), Rational(-3, 2), S.Half, pi / 2).doit() == sqrt(6) / 4
    assert Rotation.d(Rational(3, 2), Rational(-3, 2), Rational(-1, 2), pi / 2).doit() == sqrt(6) / 4
    assert Rotation.d(Rational(3, 2), Rational(-3, 2), Rational(-3, 2), pi / 2).doit() == sqrt(2) / 4
    assert Rotation.d(2, 2, 2, pi / 2).doit() == Rational(1, 4)
    assert Rotation.d(2, 2, 1, pi / 2).doit() == Rational(-1, 2)
    assert Rotation.d(2, 2, 0, pi / 2).doit() == sqrt(6) / 4
    assert Rotation.d(2, 2, -1, pi / 2).doit() == Rational(-1, 2)
    assert Rotation.d(2, 2, -2, pi / 2).doit() == Rational(1, 4)
    assert Rotation.d(2, 1, 2, pi / 2).doit() == S.Half
    assert Rotation.d(2, 1, 1, pi / 2).doit() == Rational(-1, 2)
    assert Rotation.d(2, 1, 0, pi / 2).doit() == 0
    assert Rotation.d(2, 1, -1, pi / 2).doit() == S.Half
    assert Rotation.d(2, 1, -2, pi / 2).doit() == Rational(-1, 2)
    assert Rotation.d(2, 0, 2, pi / 2).doit() == sqrt(6) / 4
    assert Rotation.d(2, 0, 1, pi / 2).doit() == 0
    assert Rotation.d(2, 0, 0, pi / 2).doit() == Rational(-1, 2)
    assert Rotation.d(2, 0, -1, pi / 2).doit() == 0
    assert Rotation.d(2, 0, -2, pi / 2).doit() == sqrt(6) / 4
    assert Rotation.d(2, -1, 2, pi / 2).doit() == S.Half
    assert Rotation.d(2, -1, 1, pi / 2).doit() == S.Half
    assert Rotation.d(2, -1, 0, pi / 2).doit() == 0
    assert Rotation.d(2, -1, -1, pi / 2).doit() == Rational(-1, 2)
    assert Rotation.d(2, -1, -2, pi / 2).doit() == Rational(-1, 2)
    assert Rotation.d(2, -2, 2, pi / 2).doit() == Rational(1, 4)
    assert Rotation.d(2, -2, 1, pi / 2).doit() == S.Half
    assert Rotation.d(2, -2, 0, pi / 2).doit() == sqrt(6) / 4
    assert Rotation.d(2, -2, -1, pi / 2).doit() == S.Half
    assert Rotation.d(2, -2, -2, pi / 2).doit() == Rational(1, 4)