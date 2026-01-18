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
def test_tan_expansion():
    assert tan(x + y).expand(trig=True) == ((tan(x) + tan(y)) / (1 - tan(x) * tan(y))).expand()
    assert tan(x - y).expand(trig=True) == ((tan(x) - tan(y)) / (1 + tan(x) * tan(y))).expand()
    assert tan(x + y + z).expand(trig=True) == ((tan(x) + tan(y) + tan(z) - tan(x) * tan(y) * tan(z)) / (1 - tan(x) * tan(y) - tan(x) * tan(z) - tan(y) * tan(z))).expand()
    assert 0 == tan(2 * x).expand(trig=True).rewrite(tan).subs([(tan(x), Rational(1, 7))]) * 24 - 7
    assert 0 == tan(3 * x).expand(trig=True).rewrite(tan).subs([(tan(x), Rational(1, 5))]) * 55 - 37
    assert 0 == tan(4 * x - pi / 4).expand(trig=True).rewrite(tan).subs([(tan(x), Rational(1, 5))]) * 239 - 1
    _test_extrig(tan, 2, 2 * tan(1) / (1 - tan(1) ** 2))
    _test_extrig(tan, 3, (-tan(1) ** 3 + 3 * tan(1)) / (1 - 3 * tan(1) ** 2))