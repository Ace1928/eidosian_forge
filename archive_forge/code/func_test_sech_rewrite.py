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
def test_sech_rewrite():
    x = Symbol('x')
    assert sech(x).rewrite(exp) == 1 / (exp(x) / 2 + exp(-x) / 2) == sech(x).rewrite('tractable')
    assert sech(x).rewrite(sinh) == I / sinh(x + I * pi / 2)
    tanh_half = tanh(S.Half * x) ** 2
    assert sech(x).rewrite(tanh) == (1 - tanh_half) / (1 + tanh_half)
    coth_half = coth(S.Half * x) ** 2
    assert sech(x).rewrite(coth) == (coth_half - 1) / (coth_half + 1)