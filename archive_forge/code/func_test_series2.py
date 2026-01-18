from sympy.calculus.util import AccumBounds
from sympy.core.function import (Derivative, PoleError)
from sympy.core.numbers import (E, I, Integer, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, acoth, asinh, atanh, cosh, coth, sinh, tanh)
from sympy.functions.elementary.integers import (ceiling, floor, frac)
from sympy.functions.elementary.miscellaneous import (cbrt, sqrt)
from sympy.functions.elementary.trigonometric import (asin, cos, cot, sin, tan)
from sympy.series.limits import limit
from sympy.series.order import O
from sympy.abc import x, y, z
from sympy.testing.pytest import raises, XFAIL
def test_series2():
    w = Symbol('w', real=True)
    x = Symbol('x', real=True)
    e = w ** (-2) * (w * exp(1 / x - w) - w * exp(1 / x))
    assert e.nseries(w, n=4) == -exp(1 / x) + w * exp(1 / x) / 2 - w ** 2 * exp(1 / x) / 6 + w ** 3 * exp(1 / x) / 24 + O(w ** 4)