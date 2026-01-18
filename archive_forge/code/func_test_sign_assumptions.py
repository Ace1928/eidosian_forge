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
def test_sign_assumptions():
    p = Symbol('p', positive=True)
    n = Symbol('n', negative=True)
    assert sinh(n).is_negative is True
    assert sinh(p).is_positive is True
    assert cosh(n).is_positive is True
    assert cosh(p).is_positive is True
    assert tanh(n).is_negative is True
    assert tanh(p).is_positive is True
    assert csch(n).is_negative is True
    assert csch(p).is_positive is True
    assert sech(n).is_positive is True
    assert sech(p).is_positive is True
    assert coth(n).is_negative is True
    assert coth(p).is_positive is True