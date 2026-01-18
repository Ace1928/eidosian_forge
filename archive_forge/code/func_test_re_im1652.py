from sympy.core.function import expand_complex
from sympy.core.numbers import (I, Integer, Rational, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (Abs, conjugate, im, re, sign)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.hyperbolic import (cosh, coth, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, cot, sin, tan)
def test_re_im1652():
    x = Symbol('x')
    assert re(x) == re(conjugate(x))
    assert im(x) == -im(conjugate(x))
    assert im(x) * re(conjugate(x)) + im(conjugate(x)) * re(x) == 0