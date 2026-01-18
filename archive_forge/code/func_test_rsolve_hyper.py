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
def test_rsolve_hyper():
    assert rsolve_hyper([-1, -1, 1], 0, n) in [C0 * (S.Half - S.Half * sqrt(5)) ** n + C1 * (S.Half + S.Half * sqrt(5)) ** n, C1 * (S.Half - S.Half * sqrt(5)) ** n + C0 * (S.Half + S.Half * sqrt(5)) ** n]
    assert rsolve_hyper([n ** 2 - 2, -2 * n - 1, 1], 0, n) in [C0 * rf(sqrt(2), n) + C1 * rf(-sqrt(2), n), C1 * rf(sqrt(2), n) + C0 * rf(-sqrt(2), n)]
    assert rsolve_hyper([n ** 2 - k, -2 * n - 1, 1], 0, n) in [C0 * rf(sqrt(k), n) + C1 * rf(-sqrt(k), n), C1 * rf(sqrt(k), n) + C0 * rf(-sqrt(k), n)]
    assert rsolve_hyper([2 * n * (n + 1), -n ** 2 - 3 * n + 2, n - 1], 0, n) == C1 * factorial(n) + C0 * 2 ** n
    assert rsolve_hyper([n + 2, -(2 * n + 3) * (17 * n ** 2 + 51 * n + 39), n + 1], 0, n) == 0
    assert rsolve_hyper([-n - 1, -1, 1], 0, n) == 0
    assert rsolve_hyper([-1, 1], n, n).expand() == C0 + n ** 2 / 2 - n / 2
    assert rsolve_hyper([-1, 1], 1 + n, n).expand() == C0 + n ** 2 / 2 + n / 2
    assert rsolve_hyper([-1, 1], 3 * (n + n ** 2), n).expand() == C0 + n ** 3 - n
    assert rsolve_hyper([-a, 1], 0, n).expand() == C0 * a ** n
    assert rsolve_hyper([-a, 0, 1], 0, n).expand() == (-1) ** n * C1 * a ** (n / 2) + C0 * a ** (n / 2)
    assert rsolve_hyper([1, 1, 1], 0, n).expand() == C0 * (Rational(-1, 2) - sqrt(3) * I / 2) ** n + C1 * (Rational(-1, 2) + sqrt(3) * I / 2) ** n
    assert rsolve_hyper([1, -2 * n / a - 2 / a, 1], 0, n) == 0