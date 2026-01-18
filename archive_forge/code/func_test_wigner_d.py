from sympy.core.numbers import (I, pi, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.spherical_harmonics import Ynm
from sympy.matrices.dense import Matrix
from sympy.physics.wigner import (clebsch_gordan, wigner_9j, wigner_6j, gaunt,
from sympy.testing.pytest import raises
def test_wigner_d():
    half = S(1) / 2
    alpha, beta, gamma = symbols('alpha, beta, gamma', real=True)
    d = wigner_d_small(half, beta).subs({beta: pi / 2})
    d_ = Matrix([[1, 1], [-1, 1]]) / sqrt(2)
    assert d == d_
    D = wigner_d(half, alpha, beta, gamma)
    assert D[0, 0] == exp(I * alpha / 2) * exp(I * gamma / 2) * cos(beta / 2)
    assert D[0, 1] == exp(I * alpha / 2) * exp(-I * gamma / 2) * sin(beta / 2)
    assert D[1, 0] == -exp(-I * alpha / 2) * exp(I * gamma / 2) * sin(beta / 2)
    assert D[1, 1] == exp(-I * alpha / 2) * exp(-I * gamma / 2) * cos(beta / 2)