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
def test_polynomial_reduce():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t ** 2, t)]})
    assert polynomial_reduce(Poly(1 + x * t + t ** 2, t), DE) == (Poly(t, t), Poly(x * t, t))
    assert polynomial_reduce(Poly(0, t), DE) == (Poly(0, t), Poly(0, t))