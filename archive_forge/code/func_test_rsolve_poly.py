from sympy.core.function import (Function, Lambda, expand)
from sympy.core.numbers import (I, Rational)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import (rf, binomial, factorial)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys.polytools import factor
from sympy.solvers.recurr import rsolve, rsolve_hyper, rsolve_poly, rsolve_ratio
from sympy.testing.pytest import raises, slow, XFAIL
from sympy.abc import a, b
def test_rsolve_poly():
    assert rsolve_poly([-1, -1, 1], 0, n) == 0
    assert rsolve_poly([-1, -1, 1], 1, n) == -1
    assert rsolve_poly([-1, n + 1], n, n) == 1
    assert rsolve_poly([-1, 1], n, n) == C0 + (n ** 2 - n) / 2
    assert rsolve_poly([-n - 1, n], 1, n) == C0 * n - 1
    assert rsolve_poly([-4 * n - 2, 1], 4 * n + 1, n) == -1
    assert rsolve_poly([-1, 1], n ** 5 + n ** 3, n) == C0 - n ** 3 / 2 - n ** 5 / 2 + n ** 2 / 6 + n ** 6 / 6 + 2 * n ** 4 / 3