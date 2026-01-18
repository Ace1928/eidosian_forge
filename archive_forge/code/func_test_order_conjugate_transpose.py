from sympy.core.add import Add
from sympy.core.function import (Function, expand)
from sympy.core.numbers import (I, Rational, nan, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (conjugate, transpose)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.series.order import O, Order
from sympy.core.expr import unchanged
from sympy.testing.pytest import raises
from sympy.abc import w, x, y, z
def test_order_conjugate_transpose():
    x = Symbol('x', real=True)
    y = Symbol('y', imaginary=True)
    assert conjugate(Order(x)) == Order(conjugate(x))
    assert conjugate(Order(y)) == Order(conjugate(y))
    assert conjugate(Order(x ** 2)) == Order(conjugate(x) ** 2)
    assert conjugate(Order(y ** 2)) == Order(conjugate(y) ** 2)
    assert transpose(Order(x)) == Order(transpose(x))
    assert transpose(Order(y)) == Order(transpose(y))
    assert transpose(Order(x ** 2)) == Order(transpose(x) ** 2)
    assert transpose(Order(y ** 2)) == Order(transpose(y) ** 2)