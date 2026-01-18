from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine
from sympy.core.expr import Expr
from sympy.core.numbers import (I, Rational, nan, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import (Abs, arg, im, re, sign)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan, atan2)
from sympy.abc import w, x, y, z
from sympy.core.relational import Eq, Ne
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices.expressions.matexpr import MatrixSymbol
def test_re():
    assert refine(re(x), Q.real(x)) == x
    assert refine(re(x), Q.imaginary(x)) is S.Zero
    assert refine(re(x + y), Q.real(x) & Q.real(y)) == x + y
    assert refine(re(x + y), Q.real(x) & Q.imaginary(y)) == x
    assert refine(re(x * y), Q.real(x) & Q.real(y)) == x * y
    assert refine(re(x * y), Q.real(x) & Q.imaginary(y)) == 0
    assert refine(re(x * y * z), Q.real(x) & Q.real(y) & Q.real(z)) == x * y * z