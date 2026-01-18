from sympy.core.function import Function
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, tan)
from sympy.testing.pytest import XFAIL
@XFAIL
def test_pow_eval_X1():
    assert (-1) ** Rational(1, 3) == S.Half + S.Half * I * sqrt(3)