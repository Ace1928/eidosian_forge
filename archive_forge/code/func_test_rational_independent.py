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
def test_rational_independent():
    ri = rational_independent
    assert ri([], x) == []
    assert ri([cos(x), sin(x)], x) == [cos(x), sin(x)]
    assert ri([x ** 2, sin(x), x * sin(x), x ** 3], x) == [x ** 3 + x ** 2, x * sin(x) + sin(x)]
    assert ri([S.One, x * log(x), log(x), sin(x) / x, cos(x), sin(x), x], x) == [x + 1, x * log(x) + log(x), sin(x) / x + sin(x), cos(x)]