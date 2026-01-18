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
def test_rational_powers_larger_than_one():
    assert Rational(2, 3) ** Rational(3, 2) == 2 * sqrt(6) / 9
    assert Rational(1, 6) ** Rational(9, 4) == 6 ** Rational(3, 4) / 216
    assert Rational(3, 7) ** Rational(7, 3) == 9 * 3 ** Rational(1, 3) * 7 ** Rational(2, 3) / 343