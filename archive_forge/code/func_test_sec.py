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
def test_sec():
    x = symbols('x', real=True)
    z = symbols('z')
    assert sec.nargs == FiniteSet(1)
    assert sec(zoo) is nan
    assert sec(0) == 1
    assert sec(pi) == -1
    assert sec(pi / 2) is zoo
    assert sec(-pi / 2) is zoo
    assert sec(pi / 6) == 2 * sqrt(3) / 3
    assert sec(pi / 3) == 2
    assert sec(pi * Rational(5, 2)) is zoo
    assert sec(pi * Rational(9, 7)) == -sec(pi * Rational(2, 7))
    assert sec(pi * Rational(3, 4)) == -sqrt(2)
    assert sec(I) == 1 / cosh(1)
    assert sec(x * I) == 1 / cosh(x)
    assert sec(-x) == sec(x)
    assert sec(asec(x)) == x
    assert sec(z).conjugate() == sec(conjugate(z))
    assert sec(z).as_real_imag() == (cos(re(z)) * cosh(im(z)) / (sin(re(z)) ** 2 * sinh(im(z)) ** 2 + cos(re(z)) ** 2 * cosh(im(z)) ** 2), sin(re(z)) * sinh(im(z)) / (sin(re(z)) ** 2 * sinh(im(z)) ** 2 + cos(re(z)) ** 2 * cosh(im(z)) ** 2))
    assert sec(x).expand(trig=True) == 1 / cos(x)
    assert sec(2 * x).expand(trig=True) == 1 / (2 * cos(x) ** 2 - 1)
    assert sec(x).is_extended_real == True
    assert sec(z).is_real == None
    assert sec(a).is_algebraic is None
    assert sec(na).is_algebraic is False
    assert sec(x).as_leading_term() == sec(x)
    assert sec(0, evaluate=False).is_finite == True
    assert sec(x).is_finite == None
    assert sec(pi / 2, evaluate=False).is_finite == False
    assert series(sec(x), x, x0=0, n=6) == 1 + x ** 2 / 2 + 5 * x ** 4 / 24 + O(x ** 6)
    assert series(sqrt(sec(x))) == 1 + x ** 2 / 4 + 7 * x ** 4 / 96 + O(x ** 6)
    assert series(sqrt(sec(x)), x, x0=pi * 3 / 2, n=4) == 1 / sqrt(x - pi * Rational(3, 2)) + (x - pi * Rational(3, 2)) ** Rational(3, 2) / 12 + (x - pi * Rational(3, 2)) ** Rational(7, 2) / 160 + O((x - pi * Rational(3, 2)) ** 4, (x, pi * Rational(3, 2)))
    assert sec(x).diff(x) == tan(x) * sec(x)
    assert sec(z).taylor_term(4, z) == 5 * z ** 4 / 24
    assert sec(z).taylor_term(6, z) == 61 * z ** 6 / 720
    assert sec(z).taylor_term(5, z) == 0