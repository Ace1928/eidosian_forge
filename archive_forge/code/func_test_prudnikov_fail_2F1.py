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
@XFAIL
def test_prudnikov_fail_2F1():
    assert can_do([a, b], [b + 1])
    assert can_do([-1, b], [c])
    assert can_do([a, b], [a + b + S.Half])
    assert can_do([a, b], [a + b - S.Half])
    assert can_do([a, b], [a + b + Rational(3, 2)])
    assert can_do([a, b], [(a + b + 1) / 2])
    assert can_do([a, b], [(a + b) / 2 + 1])
    assert can_do([a, b], [a - b + 1])
    assert can_do([a, b], [a - b + 2])
    assert can_do([a, b], [2 * b])
    assert can_do([a, b], [S.Half])
    assert can_do([a, b], [Rational(3, 2)])
    assert can_do([a, 1 - a], [c])
    assert can_do([a, 2 - a], [c])
    assert can_do([a, 3 - a], [c])
    assert can_do([a, a + S.Half], [c])
    assert can_do([1, b], [c])
    assert can_do([1, b], [Rational(3, 2)])
    assert can_do([Rational(1, 4), Rational(3, 4)], [1])
    o = S.One
    assert can_do([o / 8, 1], [o / 8 * 9])
    assert can_do([o / 6, 1], [o / 6 * 7])
    assert can_do([o / 6, 1], [o / 6 * 13])
    assert can_do([o / 5, 1], [o / 5 * 6])
    assert can_do([o / 5, 1], [o / 5 * 11])
    assert can_do([o / 4, 1], [o / 4 * 5])
    assert can_do([o / 4, 1], [o / 4 * 9])
    assert can_do([o / 3, 1], [o / 3 * 4])
    assert can_do([o / 3, 1], [o / 3 * 7])
    assert can_do([o / 8 * 3, 1], [o / 8 * 11])
    assert can_do([o / 5 * 2, 1], [o / 5 * 7])
    assert can_do([o / 5 * 2, 1], [o / 5 * 12])
    assert can_do([o / 5 * 3, 1], [o / 5 * 8])
    assert can_do([o / 5 * 3, 1], [o / 5 * 13])
    assert can_do([o / 8 * 5, 1], [o / 8 * 13])
    assert can_do([o / 4 * 3, 1], [o / 4 * 7])
    assert can_do([o / 4 * 3, 1], [o / 4 * 11])
    assert can_do([o / 3 * 2, 1], [o / 3 * 5])
    assert can_do([o / 3 * 2, 1], [o / 3 * 8])
    assert can_do([o / 5 * 4, 1], [o / 5 * 9])
    assert can_do([o / 5 * 4, 1], [o / 5 * 14])
    assert can_do([o / 6 * 5, 1], [o / 6 * 11])
    assert can_do([o / 6 * 5, 1], [o / 6 * 17])
    assert can_do([o / 8 * 7, 1], [o / 8 * 15])