from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, asech)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin)
from sympy.functions.special.bessel import airyai
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import integrate
from sympy.series.formal import fps
from sympy.series.order import O
from sympy.series.formal import (rational_algorithm, FormalPowerSeries,
from sympy.testing.pytest import raises, XFAIL, slow
def test_rational_algorithm():
    f = 1 / ((x - 1) ** 2 * (x - 2))
    assert rational_algorithm(f, x, k) == (-2 ** (-k - 1) + 1 - factorial(k + 1) / factorial(k), 0, 0)
    f = (1 + x + x ** 2 + x ** 3) / ((x - 1) * (x - 2))
    assert rational_algorithm(f, x, k) == (-15 * 2 ** (-k - 1) + 4, x + 4, 0)
    f = z / (y * m - m * x - y * x + x ** 2)
    assert rational_algorithm(f, x, k) == (-y ** (-k - 1) * z / (y - m) + m ** (-k - 1) * z / (y - m), 0, 0)
    f = x / (1 - x - x ** 2)
    assert rational_algorithm(f, x, k) is None
    assert rational_algorithm(f, x, k, full=True) == ((Rational(-1, 2) + sqrt(5) / 2) ** (-k - 1) * (-sqrt(5) / 10 + S.Half) + (-sqrt(5) / 2 - S.Half) ** (-k - 1) * (sqrt(5) / 10 + S.Half), 0, 0)
    f = 1 / (x ** 2 + 2 * x + 2)
    assert rational_algorithm(f, x, k) is None
    assert rational_algorithm(f, x, k, full=True) == (I * (-1 + I) ** (-k - 1) / 2 - I * (-1 - I) ** (-k - 1) / 2, 0, 0)
    f = log(1 + x)
    assert rational_algorithm(f, x, k) == (-(-1) ** (-k) / k, 0, 1)
    f = atan(x)
    assert rational_algorithm(f, x, k) is None
    assert rational_algorithm(f, x, k, full=True) == ((I * I ** (-k) / 2 - I * (-I) ** (-k) / 2) / k, 0, 1)
    f = x * atan(x) - log(1 + x ** 2) / 2
    assert rational_algorithm(f, x, k) is None
    assert rational_algorithm(f, x, k, full=True) == ((I * I ** (-k + 1) / 2 - I * (-I) ** (-k + 1) / 2) / (k * (k - 1)), 0, 2)
    f = log((1 + x) / (1 - x)) / 2 - atan(x)
    assert rational_algorithm(f, x, k) is None
    assert rational_algorithm(f, x, k, full=True) == ((-(-1) ** (-k) / 2 - I * I ** (-k) / 2 + I * (-I) ** (-k) / 2 + S.Half) / k, 0, 1)
    assert rational_algorithm(cos(x), x, k) is None