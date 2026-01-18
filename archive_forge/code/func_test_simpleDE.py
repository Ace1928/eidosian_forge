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
def test_simpleDE():
    for DE in simpleDE(exp(x), x, f):
        assert DE == (-f(x) + Derivative(f(x), x), 1)
        break
    for DE in simpleDE(sin(x), x, f):
        assert DE == (f(x) + Derivative(f(x), x, x), 2)
        break
    for DE in simpleDE(log(1 + x), x, f):
        assert DE == ((x + 1) * Derivative(f(x), x, 2) + Derivative(f(x), x), 2)
        break
    for DE in simpleDE(asin(x), x, f):
        assert DE == (x * Derivative(f(x), x) + (x ** 2 - 1) * Derivative(f(x), x, x), 2)
        break
    for DE in simpleDE(exp(x) * sin(x), x, f):
        assert DE == (2 * f(x) - 2 * Derivative(f(x)) + Derivative(f(x), x, x), 2)
        break
    for DE in simpleDE(((1 + x) / (1 - x)) ** n, x, f):
        assert DE == (2 * n * f(x) + (x ** 2 - 1) * Derivative(f(x), x), 1)
        break
    for DE in simpleDE(airyai(x), x, f):
        assert DE == (-x * f(x) + Derivative(f(x), x, x), 2)
        break