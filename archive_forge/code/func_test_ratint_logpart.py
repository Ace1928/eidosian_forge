from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import atan
from sympy.integrals.integrals import integrate
from sympy.polys.polytools import Poly
from sympy.simplify.simplify import simplify
from sympy.integrals.rationaltools import ratint, ratint_logpart, log_to_atan
from sympy.abc import a, b, x, t
def test_ratint_logpart():
    assert ratint_logpart(x, x ** 2 - 9, x, t) == [(Poly(x ** 2 - 9, x), Poly(-2 * t + 1, t))]
    assert ratint_logpart(x ** 2, x ** 3 - 5, x, t) == [(Poly(x ** 3 - 5, x), Poly(-3 * t + 1, t))]