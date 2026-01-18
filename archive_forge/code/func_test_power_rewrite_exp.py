from sympy.core import (
from sympy.core.parameters import global_parameters
from sympy.core.tests.test_evalf import NS
from sympy.core.function import expand_multinomial
from sympy.functions.elementary.miscellaneous import sqrt, cbrt
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.special.error_functions import erf
from sympy.functions.elementary.trigonometric import (
from sympy.functions.elementary.hyperbolic import cosh, sinh, tanh
from sympy.polys import Poly
from sympy.series.order import O
from sympy.sets import FiniteSet
from sympy.core.power import power, integer_nthroot
from sympy.testing.pytest import warns, _both_exp_pow
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.abc import a, b, c, x, y
@_both_exp_pow
def test_power_rewrite_exp():
    assert (I ** I).rewrite(exp) == exp(-pi / 2)
    expr = (2 + 3 * I) ** (4 + 5 * I)
    assert expr.rewrite(exp) == exp((4 + 5 * I) * (log(sqrt(13)) + I * atan(Rational(3, 2))))
    assert expr.rewrite(exp).expand() == 169 * exp(5 * I * log(13) / 2) * exp(4 * I * atan(Rational(3, 2))) * exp(-5 * atan(Rational(3, 2)))
    assert ((6 + 7 * I) ** 5).rewrite(exp) == 7225 * sqrt(85) * exp(5 * I * atan(Rational(7, 6)))
    expr = 5 ** (6 + 7 * I)
    assert expr.rewrite(exp) == exp((6 + 7 * I) * log(5))
    assert expr.rewrite(exp).expand() == 15625 * exp(7 * I * log(5))
    assert Pow(123, 789, evaluate=False).rewrite(exp) == 123 ** 789
    assert (1 ** I).rewrite(exp) == 1 ** I
    assert (0 ** I).rewrite(exp) == 0 ** I
    expr = (-2) ** (2 + 5 * I)
    assert expr.rewrite(exp) == exp((2 + 5 * I) * (log(2) + I * pi))
    assert expr.rewrite(exp).expand() == 4 * exp(-5 * pi) * exp(5 * I * log(2))
    assert ((-2) ** S(-5)).rewrite(exp) == (-2) ** S(-5)
    x, y = symbols('x y')
    assert (x ** y).rewrite(exp) == exp(y * log(x))
    if global_parameters.exp_is_pow:
        assert (7 ** x).rewrite(exp) == Pow(S.Exp1, x * log(7), evaluate=False)
    else:
        assert (7 ** x).rewrite(exp) == exp(x * log(7), evaluate=False)
    assert ((2 + 3 * I) ** x).rewrite(exp) == exp(x * (log(sqrt(13)) + I * atan(Rational(3, 2))))
    assert (y ** (5 + 6 * I)).rewrite(exp) == exp(log(y) * (5 + 6 * I))
    assert all(((1 / func(x)).rewrite(exp) == 1 / func(x).rewrite(exp) for func in (sin, cos, tan, sec, csc, sinh, cosh, tanh)))