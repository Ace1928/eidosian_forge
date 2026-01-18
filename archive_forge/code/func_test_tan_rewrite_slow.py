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
def test_tan_rewrite_slow():
    assert 0 == (cos(pi / 34) * tan(pi / 34) - sin(pi / 34)).rewrite(pow)
    assert 0 == (cos(pi / 17) * tan(pi / 17) - sin(pi / 17)).rewrite(pow)
    assert tan(pi / 19).rewrite(pow) == tan(pi / 19)
    assert tan(pi * Rational(8, 19)).rewrite(sqrt) == tan(pi * Rational(8, 19))
    assert tan(pi * Rational(2, 5), evaluate=False).rewrite(sqrt) == sqrt(sqrt(5) / 8 + Rational(5, 8)) / (Rational(-1, 4) + sqrt(5) / 4)