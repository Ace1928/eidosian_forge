from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import (cos, cot, sin)
from sympy.testing.pytest import _both_exp_pow
@_both_exp_pow
def test_sin_exp_rewrite():
    assert sin(x).rewrite(sin, exp) == -I / 2 * (exp(I * x) - exp(-I * x))
    assert sin(x).rewrite(sin, exp).rewrite(exp, sin) == sin(x)
    assert cos(x).rewrite(cos, exp).rewrite(exp, cos) == cos(x)
    assert (sin(5 * y) - sin(2 * x)).rewrite(sin, exp).rewrite(exp, sin) == sin(5 * y) - sin(2 * x)
    assert sin(x + y).rewrite(sin, exp).rewrite(exp, sin) == sin(x + y)
    assert cos(x + y).rewrite(cos, exp).rewrite(exp, cos) == cos(x + y)
    assert cos(x).rewrite(cos, exp).rewrite(exp, sin) == cos(x)