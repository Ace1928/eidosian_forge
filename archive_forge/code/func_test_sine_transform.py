from sympy.integrals.transforms import (
from sympy.integrals.laplace import (
from sympy.core.function import Function, expand_mul
from sympy.core import EulerGamma
from sympy.core.numbers import I, Rational, oo, pi
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import re, unpolarify
from sympy.functions.elementary.exponential import exp, exp_polar, log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import atan, cos, sin, tan
from sympy.functions.special.bessel import besseli, besselj, besselk, bessely
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.special.error_functions import erf, expint
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import meijerg
from sympy.simplify.gammasimp import gammasimp
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.trigsimp import trigsimp
from sympy.testing.pytest import XFAIL, slow, skip, raises
from sympy.abc import x, s, a, b, c, d
def test_sine_transform():
    t = symbols('t')
    w = symbols('w')
    a = symbols('a')
    f = Function('f')
    assert sine_transform(f(t), t, w) == SineTransform(f(t), t, w)
    assert inverse_sine_transform(f(w), w, t) == InverseSineTransform(f(w), w, t)
    assert sine_transform(1 / sqrt(t), t, w) == 1 / sqrt(w)
    assert inverse_sine_transform(1 / sqrt(w), w, t) == 1 / sqrt(t)
    assert sine_transform((1 / sqrt(t)) ** 3, t, w) == 2 * sqrt(w)
    assert sine_transform(t ** (-a), t, w) == 2 ** (-a + S.Half) * w ** (a - 1) * gamma(-a / 2 + 1) / gamma((a + 1) / 2)
    assert inverse_sine_transform(2 ** (-a + S(1) / 2) * w ** (a - 1) * gamma(-a / 2 + 1) / gamma(a / 2 + S.Half), w, t) == t ** (-a)
    assert sine_transform(exp(-a * t), t, w) == sqrt(2) * w / (sqrt(pi) * (a ** 2 + w ** 2))
    assert inverse_sine_transform(sqrt(2) * w / (sqrt(pi) * (a ** 2 + w ** 2)), w, t) == exp(-a * t)
    assert sine_transform(log(t) / t, t, w) == sqrt(2) * sqrt(pi) * -(log(w ** 2) + 2 * EulerGamma) / 4
    assert sine_transform(t * exp(-a * t ** 2), t, w) == sqrt(2) * w * exp(-w ** 2 / (4 * a)) / (4 * a ** Rational(3, 2))
    assert inverse_sine_transform(sqrt(2) * w * exp(-w ** 2 / (4 * a)) / (4 * a ** Rational(3, 2)), w, t) == t * exp(-a * t ** 2)