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
def test_simple_8():
    assert O(sqrt(-x)) == O(sqrt(x))
    assert O(x ** 2 * sqrt(x)) == O(x ** Rational(5, 2))
    assert O(x ** 3 * sqrt(-(-x) ** 3)) == O(x ** Rational(9, 2))
    assert O(x ** Rational(3, 2) * sqrt((-x) ** 3)) == O(x ** 3)
    assert O(x * (-2 * x) ** (I / 2)) == O(x * (-x) ** (I / 2))