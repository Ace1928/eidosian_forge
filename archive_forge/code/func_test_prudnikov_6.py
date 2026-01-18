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
def test_prudnikov_6():
    h = S.Half
    for m in [3 * h, 5 * h]:
        for n in [1, 2, 3]:
            for q in [h, 1, 2]:
                for p in [1, 2, 3]:
                    assert can_do([h, q, p], [m, n])
            for q in [1, 2, 3]:
                for p in [3 * h, 5 * h]:
                    assert can_do([h, q, p], [m, n])
    for q in [1, 2]:
        for p in [1, 2, 3]:
            for m in [1, 2, 3]:
                for n in [1, 2, 3]:
                    assert can_do([h, q, p], [m, n])
    assert can_do([h, h, 5 * h], [3 * h, 3 * h])
    assert can_do([h, 1, 5 * h], [3 * h, 3 * h])
    assert can_do([h, 2, 2], [1, 3])