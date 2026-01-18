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
def test_residue_reduce():
    a = Poly(2 * t ** 2 - t - x ** 2, t)
    d = Poly(t ** 3 - x ** 2 * t, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 / x, t)], 'Tfuncs': [log]})
    assert residue_reduce(a, d, DE, z, invert=False) == ([(Poly(z ** 2 - Rational(1, 4), z, domain='ZZ(x)'), Poly((1 + 3 * x * z - 6 * z ** 2 - 2 * x ** 2 + 4 * x ** 2 * z ** 2) * t - x * z + x ** 2 + 2 * x ** 2 * z ** 2 - 2 * z * x ** 3, t, domain='ZZ(z, x)'))], False)
    assert residue_reduce(a, d, DE, z, invert=True) == ([(Poly(z ** 2 - Rational(1, 4), z, domain='ZZ(x)'), Poly(t + 2 * x * z, t))], False)
    assert residue_reduce(Poly(-2 / x, t), Poly(t ** 2 - 1, t), DE, z, invert=False) == ([(Poly(z ** 2 - 1, z, domain='QQ'), Poly(-2 * z * t / x - 2 / x, t, domain='ZZ(z,x)'))], True)
    ans = residue_reduce(Poly(-2 / x, t), Poly(t ** 2 - 1, t), DE, z, invert=True)
    assert ans == ([(Poly(z ** 2 - 1, z, domain='QQ'), Poly(t + z, t))], True)
    assert residue_reduce_to_basic(ans[0], DE, z) == -log(-1 + log(x)) + log(1 + log(x))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-t ** 2 - t / x - (1 - nu ** 2 / x ** 2), t)]})
    assert residue_reduce(Poly((-2 * nu ** 2 - x ** 4) / (2 * x ** 2) * t - (1 + x ** 2) / x, t), Poly(t ** 2 + 1 + x ** 2 / 2, t), DE, z) == ([(Poly(z + S.Half, z, domain='QQ'), Poly(t ** 2 + 1 + x ** 2 / 2, t, domain='ZZ(x,nu)'))], True)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t ** 2, t)]})
    assert residue_reduce(Poly(-2 * x * t + 1 - x ** 2, t), Poly(t ** 2 + 2 * x * t + 1 + x ** 2, t), DE, z) == ([(Poly(z ** 2 + Rational(1, 4), z), Poly(t + x + 2 * z, t))], True)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    assert residue_reduce(Poly(t, t), Poly(t + sqrt(2), t), DE, z) == ([(Poly(z - 1, z, domain='QQ'), Poly(t + sqrt(2), t))], True)