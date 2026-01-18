from sympy.core.function import (Derivative, Function)
from sympy.core.numbers import oo
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import cos
from sympy.integrals.integrals import Integral
from sympy.functions.special.bessel import besselj
from sympy.functions.special.polynomials import legendre
from sympy.functions.combinatorial.numbers import bell
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.testing.pytest import XFAIL
@XFAIL
def test_requires_partial_unspecified_variables():
    x, y = symbols('x y')
    f = symbols('f', cls=Function)
    assert requires_partial(Derivative(f, x)) is False
    assert requires_partial(Derivative(f, x, y)) is True