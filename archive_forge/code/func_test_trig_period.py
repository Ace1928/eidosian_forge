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
def test_trig_period():
    x, y = symbols('x, y')
    assert sin(x).period() == 2 * pi
    assert cos(x).period() == 2 * pi
    assert tan(x).period() == pi
    assert cot(x).period() == pi
    assert sec(x).period() == 2 * pi
    assert csc(x).period() == 2 * pi
    assert sin(2 * x).period() == pi
    assert cot(4 * x - 6).period() == pi / 4
    assert cos(-3 * x).period() == pi * Rational(2, 3)
    assert cos(x * y).period(x) == 2 * pi / abs(y)
    assert sin(3 * x * y + 2 * pi).period(y) == 2 * pi / abs(3 * x)
    assert tan(3 * x).period(y) is S.Zero
    raises(NotImplementedError, lambda: sin(x ** 2).period(x))