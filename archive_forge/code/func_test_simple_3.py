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
def test_simple_3():
    assert Order(x) + x == Order(x)
    assert Order(x) + 2 == 2 + Order(x)
    assert Order(x) + x ** 2 == Order(x)
    assert Order(x) + 1 / x == 1 / x + Order(x)
    assert Order(1 / x) + 1 / x ** 2 == 1 / x ** 2 + Order(1 / x)
    assert Order(x) + exp(1 / x) == Order(x) + exp(1 / x)