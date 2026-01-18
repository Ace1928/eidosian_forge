from sympy.core.function import (Function, Lambda, diff, expand_log)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan, sin, tan)
from sympy.polys.polytools import (Poly, cancel, factor)
from sympy.integrals.risch import (gcdex_diophantine, frac_in, as_poly_1t,
from sympy.testing.pytest import raises
from sympy.abc import x, t, nu, z, a, y
def test_splitfactor():
    p = Poly(4 * x ** 4 * t ** 5 + (-4 * x ** 3 - 4 * x ** 4) * t ** 4 + (-3 * x ** 2 + 2 * x ** 3) * t ** 3 + (2 * x + 7 * x ** 2 + 2 * x ** 3) * t ** 2 + (1 - 4 * x - 4 * x ** 2) * t - 1 + 2 * x, t, field=True)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-t ** 2 - 3 / (2 * x) * t + 1 / (2 * x), t)]})
    assert splitfactor(p, DE) == (Poly(4 * x ** 4 * t ** 3 + (-8 * x ** 3 - 4 * x ** 4) * t ** 2 + (4 * x ** 2 + 8 * x ** 3) * t - 4 * x ** 2, t, domain='ZZ(x)'), Poly(t ** 2 + 1 / x * t + (1 - 2 * x) / (4 * x ** 2), t, domain='ZZ(x)'))
    assert splitfactor(Poly(x, t), DE) == (Poly(x, t), Poly(1, t))
    r = Poly(-4 * x ** 4 * z ** 2 + 4 * x ** 6 * z ** 2 - z * x ** 3 - 4 * x ** 5 * z ** 3 + 4 * x ** 3 * z ** 3 + x ** 4 + z * x ** 5 - x ** 6, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 / x, t)]})
    assert splitfactor(r, DE, coefficientD=True) == (Poly(x * z - x ** 2 - z * x ** 3 + x ** 4, t), Poly(-x ** 2 + 4 * x ** 2 * z ** 2, t))
    assert splitfactor_sqf(r, DE, coefficientD=True) == (((Poly(x * z - x ** 2 - z * x ** 3 + x ** 4, t), 1),), ((Poly(-x ** 2 + 4 * x ** 2 * z ** 2, t), 1),))
    assert splitfactor(Poly(0, t), DE) == (Poly(0, t), Poly(1, t))
    assert splitfactor_sqf(Poly(0, t), DE) == (((Poly(0, t), 1),), ())