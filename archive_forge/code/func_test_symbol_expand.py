from sympy.core.function import Function
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, tan)
from sympy.testing.pytest import XFAIL
def test_symbol_expand():
    x = Symbol('x')
    y = Symbol('y')
    f = x ** 4 * y ** 4
    assert f == x ** 4 * y ** 4
    assert f == f.expand()
    g = (x * y) ** 4
    assert g == f
    assert g.expand() == f
    assert g.expand() == g.expand().expand()