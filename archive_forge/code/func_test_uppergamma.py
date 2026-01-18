from sympy.core.function import expand_func, Subs
from sympy.core import EulerGamma
from sympy.core.numbers import (I, Rational, nan, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.combinatorial.numbers import harmonic
from sympy.functions.elementary.complexes import (Abs, conjugate, im, re)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, atan)
from sympy.functions.special.error_functions import (Ei, erf, erfc)
from sympy.functions.special.gamma_functions import (digamma, gamma, loggamma, lowergamma, multigamma, polygamma, trigamma, uppergamma)
from sympy.functions.special.zeta_functions import zeta
from sympy.series.order import O
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import raises
from sympy.core.random import (test_derivative_numerically as td,
def test_uppergamma():
    from sympy.functions.special.error_functions import expint
    from sympy.functions.special.hyper import meijerg
    assert uppergamma(4, 0) == 6
    assert uppergamma(x, y).diff(y) == -y ** (x - 1) * exp(-y)
    assert td(uppergamma(randcplx(), y), y)
    assert uppergamma(x, y).diff(x) == uppergamma(x, y) * log(y) + meijerg([], [1, 1], [0, 0, x], [], y)
    assert td(uppergamma(x, randcplx()), x)
    p = Symbol('p', positive=True)
    assert uppergamma(0, p) == -Ei(-p)
    assert uppergamma(p, 0) == gamma(p)
    assert uppergamma(S.Half, x) == sqrt(pi) * erfc(sqrt(x))
    assert not uppergamma(S.Half - 3, x).has(uppergamma)
    assert not uppergamma(S.Half + 3, x).has(uppergamma)
    assert uppergamma(S.Half, x, evaluate=False).has(uppergamma)
    assert tn(uppergamma(S.Half + 3, x, evaluate=False), uppergamma(S.Half + 3, x), x)
    assert tn(uppergamma(S.Half - 3, x, evaluate=False), uppergamma(S.Half - 3, x), x)
    assert unchanged(uppergamma, x, -oo)
    assert unchanged(uppergamma, x, 0)
    assert tn_branch(-3, uppergamma)
    assert tn_branch(-4, uppergamma)
    assert tn_branch(Rational(1, 3), uppergamma)
    assert tn_branch(pi, uppergamma)
    assert uppergamma(3, exp_polar(4 * pi * I) * x) == uppergamma(3, x)
    assert uppergamma(y, exp_polar(5 * pi * I) * x) == exp(4 * I * pi * y) * uppergamma(y, x * exp_polar(pi * I)) + gamma(y) * (1 - exp(4 * pi * I * y))
    assert uppergamma(-2, exp_polar(5 * pi * I) * x) == uppergamma(-2, x * exp_polar(I * pi)) - 2 * pi * I
    assert uppergamma(-2, x) == expint(3, x) / x ** 2
    assert conjugate(uppergamma(x, y)) == uppergamma(conjugate(x), conjugate(y))
    assert unchanged(conjugate, uppergamma(x, -oo))
    assert uppergamma(x, y).rewrite(expint) == y ** x * expint(-x + 1, y)
    assert uppergamma(x, y).rewrite(lowergamma) == gamma(x) - lowergamma(x, y)
    assert uppergamma(70, 6) == 69035724522603011058660187038367026272747334489677105069435923032634389419656200387949342530805432320 * exp(-6)
    assert (uppergamma(S(77) / 2, 6) - uppergamma(S(77) / 2, 6, evaluate=False)).evalf() < 1e-16
    assert (uppergamma(-S(77) / 2, 6) - uppergamma(-S(77) / 2, 6, evaluate=False)).evalf() < 1e-16