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
def test_shift_operators():
    a1, a2, b1, b2, b3 = (randcplx(n) for n in range(5))
    h = hyper((a1, a2), (b1, b2, b3), z)
    raises(ValueError, lambda: ShiftA(0))
    raises(ValueError, lambda: ShiftB(1))
    assert tn(ShiftA(a1).apply(h, op), hyper((a1 + 1, a2), (b1, b2, b3), z), z)
    assert tn(ShiftA(a2).apply(h, op), hyper((a1, a2 + 1), (b1, b2, b3), z), z)
    assert tn(ShiftB(b1).apply(h, op), hyper((a1, a2), (b1 - 1, b2, b3), z), z)
    assert tn(ShiftB(b2).apply(h, op), hyper((a1, a2), (b1, b2 - 1, b3), z), z)
    assert tn(ShiftB(b3).apply(h, op), hyper((a1, a2), (b1, b2, b3 - 1), z), z)