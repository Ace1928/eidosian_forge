from sympy.core.random import randrange
from sympy.simplify.hyperexpand import (ShiftA, ShiftB, UnShiftA, UnShiftB,
from sympy.concrete.summations import Sum
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.abc import z, a, b, c
from sympy.testing.pytest import XFAIL, raises, slow, ON_CI, skip
from sympy.core.random import verify_numerically as tn
from sympy.core.numbers import (Rational, pi)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import atanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)
from sympy.functions.special.bessel import besseli
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import (gamma, lowergamma)
@slow
def test_prudnikov_misc():
    assert can_do([1, (3 + I) / 2, (3 - I) / 2], [Rational(3, 2), 2])
    assert can_do([S.Half, a - 1], [Rational(3, 2), a + 1], lowerplane=True)
    assert can_do([], [b + 1])
    assert can_do([a], [a - 1, b + 1])
    assert can_do([a], [a - S.Half, 2 * a])
    assert can_do([a], [a - S.Half, 2 * a + 1])
    assert can_do([a], [a - S.Half, 2 * a - 1])
    assert can_do([a], [a + S.Half, 2 * a])
    assert can_do([a], [a + S.Half, 2 * a + 1])
    assert can_do([a], [a + S.Half, 2 * a - 1])
    assert can_do([S.Half], [b, 2 - b])
    assert can_do([S.Half], [b, 3 - b])
    assert can_do([1], [2, b])
    assert can_do([a, a + S.Half], [2 * a, b, 2 * a - b + 1])
    assert can_do([a, a + S.Half], [S.Half, 2 * a, 2 * a + S.Half])
    assert can_do([a], [a + 1], lowerplane=True)