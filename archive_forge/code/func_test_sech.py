from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.function import (expand_mul, expand_trig)
from sympy.core.numbers import (E, I, Integer, Rational, nan, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, acoth, acsch, asech, asinh, atanh, cosh, coth, csch, sech, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, cos, cot, sec, sin, tan)
from sympy.series.order import O
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import raises
def test_sech():
    x, y = symbols('x, y')
    k = Symbol('k', integer=True)
    n = Symbol('n', positive=True)
    assert sech(nan) is nan
    assert sech(zoo) is nan
    assert sech(oo) == 0
    assert sech(-oo) == 0
    assert sech(0) == 1
    assert sech(-1) == sech(1)
    assert sech(-x) == sech(x)
    assert sech(pi * I) == sec(pi)
    assert sech(-pi * I) == sec(pi)
    assert sech(-2 ** 1024 * E) == sech(2 ** 1024 * E)
    assert sech(pi * I / 2) is zoo
    assert sech(-pi * I / 2) is zoo
    assert sech((-3 * 10 ** 73 + 1) * pi * I / 2) is zoo
    assert sech((7 * 10 ** 103 + 1) * pi * I / 2) is zoo
    assert sech(pi * I) == -1
    assert sech(-pi * I) == -1
    assert sech(5 * pi * I) == -1
    assert sech(8 * pi * I) == 1
    assert sech(pi * I / 3) == 2
    assert sech(pi * I * Rational(-2, 3)) == -2
    assert sech(pi * I / 4) == sqrt(2)
    assert sech(-pi * I / 4) == sqrt(2)
    assert sech(pi * I * Rational(5, 4)) == -sqrt(2)
    assert sech(pi * I * Rational(-5, 4)) == -sqrt(2)
    assert sech(pi * I / 6) == 2 / sqrt(3)
    assert sech(-pi * I / 6) == 2 / sqrt(3)
    assert sech(pi * I * Rational(7, 6)) == -2 / sqrt(3)
    assert sech(pi * I * Rational(-5, 6)) == -2 / sqrt(3)
    assert sech(pi * I / 105) == 1 / cos(pi / 105)
    assert sech(-pi * I / 105) == 1 / cos(pi / 105)
    assert sech(x * I) == 1 / cos(x)
    assert sech(k * pi * I) == 1 / cos(k * pi)
    assert sech(17 * k * pi * I) == 1 / cos(17 * k * pi)
    assert sech(n).is_real is True
    assert expand_trig(sech(x + y)) == 1 / (cosh(x) * cosh(y) + sinh(x) * sinh(y))