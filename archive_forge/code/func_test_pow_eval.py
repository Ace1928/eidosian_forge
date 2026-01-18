from sympy.core.function import Function
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, tan)
from sympy.testing.pytest import XFAIL
def test_pow_eval():
    assert sqrt(-1) == I
    assert sqrt(-4) == 2 * I
    assert sqrt(4) == 2
    assert 8 ** Rational(1, 3) == 2
    assert (-8) ** Rational(1, 3) == 2 * (-1) ** Rational(1, 3)
    assert sqrt(-2) == I * sqrt(2)
    assert (-1) ** Rational(1, 3) != I
    assert (-10) ** Rational(1, 3) != I * 10 ** Rational(1, 3)
    assert (-2) ** Rational(1, 4) != 2 ** Rational(1, 4)
    assert 64 ** Rational(1, 3) == 4
    assert 64 ** Rational(2, 3) == 16
    assert 24 / sqrt(64) == 3
    assert (-27) ** Rational(1, 3) == 3 * (-1) ** Rational(1, 3)
    assert (cos(2) / tan(2)) ** 2 == (cos(2) / tan(2)) ** 2