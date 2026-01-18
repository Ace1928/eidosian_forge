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
def test_rsolve():
    f = y(n + 2) - y(n + 1) - y(n)
    h = sqrt(5) * (S.Half + S.Half * sqrt(5)) ** n - sqrt(5) * (S.Half - S.Half * sqrt(5)) ** n
    assert rsolve(f, y(n)) in [C0 * (S.Half - S.Half * sqrt(5)) ** n + C1 * (S.Half + S.Half * sqrt(5)) ** n, C1 * (S.Half - S.Half * sqrt(5)) ** n + C0 * (S.Half + S.Half * sqrt(5)) ** n]
    assert rsolve(f, y(n), [0, 5]) == h
    assert rsolve(f, y(n), {0: 0, 1: 5}) == h
    assert rsolve(f, y(n), {y(0): 0, y(1): 5}) == h
    assert rsolve(y(n) - y(n - 1) - y(n - 2), y(n), [0, 5]) == h
    assert rsolve(Eq(y(n), y(n - 1) + y(n - 2)), y(n), [0, 5]) == h
    assert f.subs(y, Lambda(k, rsolve(f, y(n)).subs(n, k))).simplify() == 0
    f = (n - 1) * y(n + 2) - (n ** 2 + 3 * n - 2) * y(n + 1) + 2 * n * (n + 1) * y(n)
    g = C1 * factorial(n) + C0 * 2 ** n
    h = -3 * factorial(n) + 3 * 2 ** n
    assert rsolve(f, y(n)) == g
    assert rsolve(f, y(n), []) == g
    assert rsolve(f, y(n), {}) == g
    assert rsolve(f, y(n), [0, 3]) == h
    assert rsolve(f, y(n), {0: 0, 1: 3}) == h
    assert rsolve(f, y(n), {y(0): 0, y(1): 3}) == h
    assert f.subs(y, Lambda(k, rsolve(f, y(n)).subs(n, k))).simplify() == 0
    f = y(n) - y(n - 1) - 2
    assert rsolve(f, y(n), {y(0): 0}) == 2 * n
    assert rsolve(f, y(n), {y(0): 1}) == 2 * n + 1
    assert rsolve(f, y(n), {y(0): 0, y(1): 1}) is None
    assert f.subs(y, Lambda(k, rsolve(f, y(n)).subs(n, k))).simplify() == 0
    f = 3 * y(n - 1) - y(n) - 1
    assert rsolve(f, y(n), {y(0): 0}) == -3 ** n / 2 + S.Half
    assert rsolve(f, y(n), {y(0): 1}) == 3 ** n / 2 + S.Half
    assert rsolve(f, y(n), {y(0): 2}) == 3 * 3 ** n / 2 + S.Half
    assert f.subs(y, Lambda(k, rsolve(f, y(n)).subs(n, k))).simplify() == 0
    f = y(n) - 1 / n * y(n - 1)
    assert rsolve(f, y(n)) == C0 / factorial(n)
    assert f.subs(y, Lambda(k, rsolve(f, y(n)).subs(n, k))).simplify() == 0
    f = y(n) - 1 / n * y(n - 1) - 1
    assert rsolve(f, y(n)) is None
    f = 2 * y(n - 1) + (1 - n) * y(n) / n
    assert rsolve(f, y(n), {y(1): 1}) == 2 ** (n - 1) * n
    assert rsolve(f, y(n), {y(1): 2}) == 2 ** (n - 1) * n * 2
    assert rsolve(f, y(n), {y(1): 3}) == 2 ** (n - 1) * n * 3
    assert f.subs(y, Lambda(k, rsolve(f, y(n)).subs(n, k))).simplify() == 0
    f = (n - 1) * (n - 2) * y(n + 2) - (n + 1) * (n + 2) * y(n)
    assert rsolve(f, y(n), {y(3): 6, y(4): 24}) == n * (n - 1) * (n - 2)
    assert rsolve(f, y(n), {y(3): 6, y(4): -24}) == -n * (n - 1) * (n - 2) * (-1) ** n
    assert f.subs(y, Lambda(k, rsolve(f, y(n)).subs(n, k))).simplify() == 0
    assert rsolve(Eq(y(n + 1), a * y(n)), y(n), {y(1): a}).simplify() == a ** n
    assert rsolve(y(n) - a * y(n - 2), y(n), {y(1): sqrt(a) * (a + b), y(2): a * (a - b)}).simplify() == a ** (n / 2 + 1) - b * (-sqrt(a)) ** n
    f = (-16 * n ** 2 + 32 * n - 12) * y(n - 1) + (4 * n ** 2 - 12 * n + 9) * y(n)
    yn = rsolve(f, y(n), {y(1): binomial(2 * n + 1, 3)})
    sol = 2 ** (2 * n) * n * (2 * n - 1) ** 2 * (2 * n + 1) / 12
    assert factor(expand(yn, func=True)) == sol
    sol = rsolve(y(n) + a * (y(n + 1) + y(n - 1)) / 2, y(n))
    assert str(sol) == 'C0*((-sqrt(1 - a**2) - 1)/a)**n + C1*((sqrt(1 - a**2) - 1)/a)**n'
    assert rsolve((k + 1) * y(k), y(k)) is None
    assert rsolve((k + 1) * y(k) + (k + 3) * y(k + 1) + (k + 5) * y(k + 2), y(k)) is None
    assert rsolve(y(n) + y(n + 1) + 2 ** n + 3 ** n, y(n)) == (-1) ** n * C0 - 2 ** n / 3 - 3 ** n / 4