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
def test_pow_issue_1724():
    e = S.NegativeOne ** (S.One / 3)
    assert e.conjugate().n() == e.n().conjugate()
    e = S('-2/3 - (-29/54 + sqrt(93)/18)**(1/3) - 1/(9*(-29/54 + sqrt(93)/18)**(1/3))')
    assert e.conjugate().n() == e.n().conjugate()
    e = 2 ** I
    assert e.conjugate().n() == e.n().conjugate()