from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.add import Add
from sympy.core.function import (Lambda, diff)
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (arg, conjugate, im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acoth, asinh, atanh, cosh, coth, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, acot, acsc, asec, asin, atan, atan2,
from sympy.functions.special.bessel import (besselj, jn)
from sympy.functions.special.delta_functions import Heaviside
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import (cancel, gcd)
from sympy.series.limits import limit
from sympy.series.order import O
from sympy.series.series import series
from sympy.sets.fancysets import ImageSet
from sympy.sets.sets import (FiniteSet, Interval)
from sympy.simplify.simplify import simplify
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.core.relational import Ne, Eq
from sympy.functions.elementary.piecewise import Piecewise
from sympy.sets.setexpr import SetExpr
from sympy.testing.pytest import XFAIL, slow, raises
@slow
def test_tancot_rewrite_sqrt():
    for p in [1, 3, 5, 17]:
        for t in [1, 8]:
            n = t * p
            for i in range(1, min((n + 1) // 2 + 1, 10)):
                if 1 == gcd(i, n):
                    x = i * pi / n
                    if 2 * i != n and 3 * i != 2 * n:
                        t1 = tan(x).rewrite(sqrt)
                        assert not t1.has(cot, tan), 'fails for %d*pi/%d' % (i, n)
                        assert 0.001 > abs(tan(x.evalf(7)) - t1.evalf(4)), 'fails for %d*pi/%d' % (i, n)
                    if i != 0 and i != n:
                        c1 = cot(x).rewrite(sqrt)
                        assert not c1.has(cot, tan), 'fails for %d*pi/%d' % (i, n)
                        assert 0.001 > abs(cot(x.evalf(7)) - c1.evalf(4)), 'fails for %d*pi/%d' % (i, n)