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
def test_sin_rewrite():
    assert sin(x).rewrite(exp) == -I * (exp(I * x) - exp(-I * x)) / 2
    assert sin(x).rewrite(tan) == 2 * tan(x / 2) / (1 + tan(x / 2) ** 2)
    assert sin(x).rewrite(cot) == Piecewise((0, Eq(im(x), 0) & Eq(Mod(x, pi), 0)), (2 * cot(x / 2) / (cot(x / 2) ** 2 + 1), True))
    assert sin(sinh(x)).rewrite(exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, sinh(3)).n()
    assert sin(cosh(x)).rewrite(exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, cosh(3)).n()
    assert sin(tanh(x)).rewrite(exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, tanh(3)).n()
    assert sin(coth(x)).rewrite(exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, coth(3)).n()
    assert sin(sin(x)).rewrite(exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, sin(3)).n()
    assert sin(cos(x)).rewrite(exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, cos(3)).n()
    assert sin(tan(x)).rewrite(exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, tan(3)).n()
    assert sin(cot(x)).rewrite(exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, cot(3)).n()
    assert sin(log(x)).rewrite(Pow) == I * x ** (-I) / 2 - I * x ** I / 2
    assert sin(x).rewrite(csc) == 1 / csc(x)
    assert sin(x).rewrite(cos) == cos(x - pi / 2, evaluate=False)
    assert sin(x).rewrite(sec) == 1 / sec(x - pi / 2, evaluate=False)
    assert sin(cos(x)).rewrite(Pow) == sin(cos(x))