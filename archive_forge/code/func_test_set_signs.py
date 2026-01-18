from itertools import product
from sympy.concrete.summations import Sum
from sympy.core.function import (Function, diff)
from sympy.core import EulerGamma
from sympy.core.numbers import (E, I, Rational, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import (binomial, factorial, subfactorial)
from sympy.functions.elementary.complexes import (Abs, re, sign)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, acoth, acsch, asech, atanh, sinh, tanh)
from sympy.functions.elementary.integers import (ceiling, floor, frac)
from sympy.functions.elementary.miscellaneous import (cbrt, real_root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, acot, acsc, asec, asin,
from sympy.functions.special.bessel import (besseli, bessely, besselj, besselk)
from sympy.functions.special.error_functions import (Ei, erf, erfc, erfi, fresnelc, fresnels)
from sympy.functions.special.gamma_functions import (digamma, gamma, uppergamma)
from sympy.functions.special.hyper import meijerg
from sympy.integrals.integrals import (Integral, integrate)
from sympy.series.limits import (Limit, limit)
from sympy.simplify.simplify import (logcombine, simplify)
from sympy.simplify.hyperexpand import hyperexpand
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.mul import Mul
from sympy.series.limits import heuristics
from sympy.series.order import Order
from sympy.testing.pytest import XFAIL, raises
from sympy.abc import x, y, z, k
def test_set_signs():
    assert limit(abs(x), x, 0) == 0
    assert limit(abs(sin(x)), x, 0) == 0
    assert limit(abs(cos(x)), x, 0) == 1
    assert limit(abs(sin(x + 1)), x, 0) == sin(1)
    assert limit((Abs(x + y) - Abs(x - y)) / (2 * x), x, 0) == sign(y)
    assert limit(Abs(log(x) / x ** 3), x, oo) == 0
    assert limit(x * (Abs(log(x) / x ** 3) / Abs(log(x + 1) / (x + 1) ** 3) - 1), x, oo) == 3
    assert limit(Abs(log(x - 1) ** 3 - 1), x, 1, '+') == oo
    assert limit(Abs(log(x)), x, 0) == oo
    assert limit(Abs(log(Abs(x))), x, 0) == oo
    z = Symbol('z', positive=True)
    assert limit(Abs(log(z) + 1) / log(z), z, oo) == 1
    assert limit(z * (Abs(1 / z + y) - Abs(y - 1 / z)) / 2, z, 0) == 0
    assert limit(cos(z) / sign(z), z, pi, '-') == -1