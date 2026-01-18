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
def test_trigamma_expand_func():
    assert trigamma(2 * x).expand(func=True) == polygamma(1, x) / 4 + polygamma(1, Rational(1, 2) + x) / 4
    assert trigamma(1 + x).expand(func=True) == polygamma(1, x) - 1 / x ** 2
    assert trigamma(2 + x).expand(func=True, multinomial=False) == polygamma(1, x) - 1 / x ** 2 - 1 / (1 + x) ** 2
    assert trigamma(3 + x).expand(func=True, multinomial=False) == polygamma(1, x) - 1 / x ** 2 - 1 / (1 + x) ** 2 - 1 / (2 + x) ** 2
    assert trigamma(4 + x).expand(func=True, multinomial=False) == polygamma(1, x) - 1 / x ** 2 - 1 / (1 + x) ** 2 - 1 / (2 + x) ** 2 - 1 / (3 + x) ** 2
    assert trigamma(x + y).expand(func=True) == polygamma(1, x + y)
    assert trigamma(3 + 4 * x + y).expand(func=True, multinomial=False) == polygamma(1, y + 4 * x) - 1 / (y + 4 * x) ** 2 - 1 / (1 + y + 4 * x) ** 2 - 1 / (2 + y + 4 * x) ** 2