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
def test_sinh_expansion():
    x, y = symbols('x,y')
    assert sinh(x + y).expand(trig=True) == sinh(x) * cosh(y) + cosh(x) * sinh(y)
    assert sinh(2 * x).expand(trig=True) == 2 * sinh(x) * cosh(x)
    assert sinh(3 * x).expand(trig=True).expand() == sinh(x) ** 3 + 3 * sinh(x) * cosh(x) ** 2