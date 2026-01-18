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
def test_partial_simp():
    a, b, c, d, e = (randcplx() for _ in range(5))
    for func in [Hyper_Function([a, b, c], [d, e]), Hyper_Function([], [a, b, c, d, e])]:
        f = build_hypergeometric_formula(func)
        z = f.z
        assert f.closed_form == func(z)
        deriv1 = f.B.diff(z) * z
        deriv2 = f.M * f.B
        for func1, func2 in zip(deriv1, deriv2):
            assert tn(func1, func2, z)
    a, b, z = symbols('a b z')
    assert hyperexpand(hyper([3, a], [1, b], z)) == (-a * b / 2 + a * z / 2 + 2 * a) * hyper([a + 1], [b], z) + (a * b / 2 - 2 * a + 1) * hyper([a], [b], z)
    assert tn(hyperexpand(hyper([3, d], [1, e], z)), hyper([3, d], [1, e], z), z)
    assert hyperexpand(hyper([3], [1, a, b], z)) == hyper((), (a, b), z) + z * hyper((), (a + 1, b), z) / (2 * a) - z * (b - 4) * hyper((), (a + 1, b + 1), z) / (2 * a * b)
    assert tn(hyperexpand(hyper([3], [1, d, e], z)), hyper([3], [1, d, e], z), z)