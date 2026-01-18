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
def test_plan_derivatives():
    a1, a2, a3 = (1, 2, S('1/2'))
    b1, b2 = (3, S('5/2'))
    h = Hyper_Function((a1, a2, a3), (b1, b2))
    h2 = Hyper_Function((a1 + 1, a2 + 1, a3 + 2), (b1 + 1, b2 + 1))
    ops = devise_plan(h2, h, z)
    f = Formula(h, z, h(z), [])
    deriv = make_derivative_operator(f.M, z)
    assert tn((apply_operators(f.C, ops, deriv) * f.B)[0], h2(z), z)
    h2 = Hyper_Function((a1, a2 - 1, a3 - 2), (b1 - 1, b2 - 1))
    ops = devise_plan(h2, h, z)
    assert tn((apply_operators(f.C, ops, deriv) * f.B)[0], h2(z), z)