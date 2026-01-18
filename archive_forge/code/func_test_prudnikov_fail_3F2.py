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
def test_prudnikov_fail_3F2():
    assert can_do([a, a + Rational(1, 3), a + Rational(2, 3)], [Rational(1, 3), Rational(2, 3)])
    assert can_do([a, a + Rational(1, 3), a + Rational(2, 3)], [Rational(2, 3), Rational(4, 3)])
    assert can_do([a, a + Rational(1, 3), a + Rational(2, 3)], [Rational(4, 3), Rational(5, 3)])
    assert can_do([a, a + Rational(1, 3), a + Rational(2, 3)], [a * Rational(3, 2), (3 * a + 1) / 2])
    assert can_do([Rational(-1, 2), S.Half, S.Half], [1, 1])
    assert can_do([Rational(-1, 2), S.Half, 1], [Rational(3, 2), Rational(3, 2)])
    assert can_do([Rational(1, 8), Rational(3, 8), 1], [Rational(9, 8), Rational(11, 8)])
    assert can_do([Rational(1, 8), Rational(5, 8), 1], [Rational(9, 8), Rational(13, 8)])
    assert can_do([Rational(1, 8), Rational(7, 8), 1], [Rational(9, 8), Rational(15, 8)])
    assert can_do([Rational(1, 6), Rational(1, 3), 1], [Rational(7, 6), Rational(4, 3)])
    assert can_do([Rational(1, 6), Rational(2, 3), 1], [Rational(7, 6), Rational(5, 3)])
    assert can_do([Rational(1, 6), Rational(2, 3), 1], [Rational(5, 3), Rational(13, 6)])
    assert can_do([S.Half, 1, 1], [Rational(1, 4), Rational(3, 4)])