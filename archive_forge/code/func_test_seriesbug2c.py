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
def test_seriesbug2c():
    w = Symbol('w', real=True)
    e = (sin(2 * w) / w) ** (1 + w)
    assert e.series(w, 0, 1) == 2 + O(w)
    assert e.series(w, 0, 3) == 2 + 2 * w * log(2) + w ** 2 * (Rational(-4, 3) + log(2) ** 2) + O(w ** 3)
    assert e.series(w, 0, 2).subs(w, 0) == 2