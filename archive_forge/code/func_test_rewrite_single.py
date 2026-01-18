from sympy.core.function import expand_func
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.functions.elementary.complexes import Abs, arg, re, unpolarify
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import cosh, acosh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
from sympy.functions.elementary.trigonometric import (cos, sin, sinc, asin)
from sympy.functions.special.error_functions import (erf, erfc)
from sympy.functions.special.gamma_functions import (gamma, polygamma)
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.integrals.integrals import (Integral, integrate)
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.simplify import simplify
from sympy.integrals.meijerint import (_rewrite_single, _rewrite1,
from sympy.testing.pytest import slow
from sympy.core.random import (verify_numerically,
from sympy.abc import x, y, a, b, c, d, s, t, z
def test_rewrite_single():

    def t(expr, c, m):
        e = _rewrite_single(meijerg([a], [b], [c], [d], expr), x)
        assert e is not None
        assert isinstance(e[0][0][2], meijerg)
        assert e[0][0][2].argument.as_coeff_mul(x) == (c, (m,))

    def tn(expr):
        assert _rewrite_single(meijerg([a], [b], [c], [d], expr), x) is None
    t(x, 1, x)
    t(x ** 2, 1, x ** 2)
    t(x ** 2 + y * x ** 2, y + 1, x ** 2)
    tn(x ** 2 + x)
    tn(x ** y)

    def u(expr, x):
        from sympy.core.add import Add
        r = _rewrite_single(expr, x)
        e = Add(*[res[0] * res[2] for res in r[0]]).replace(exp_polar, exp)
        assert verify_numerically(e, expr, x)
    u(exp(-x) * sin(x), x)
    assert _rewrite_single(exp(x) * sin(x), x) == ([(-sqrt(2) / (2 * sqrt(pi)), 0, meijerg(((Rational(-1, 2), 0, Rational(1, 4), S.Half, Rational(3, 4)), (1,)), ((), (Rational(-1, 2), 0)), 64 * exp_polar(-4 * I * pi) / x ** 4))], True)