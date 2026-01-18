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
def test_rsolve_bulk():
    """Some bulk-generated tests."""
    funcs = [n, n + 1, n ** 2, n ** 3, n ** 4, n + n ** 2, 27 * n + 52 * n ** 2 - 3 * n ** 3 + 12 * n ** 4 - 52 * n ** 5]
    coeffs = [[-2, 1], [-2, -1, 1], [-1, 1, 1, -1, 1], [-n, 1], [n ** 2 - n + 12, 1]]
    for p in funcs:
        for c in coeffs:
            q = recurrence_term(c, p)
            if p.is_polynomial(n):
                assert rsolve_poly(c, q, n) == p
            if p.is_hypergeometric(n) and len(c) <= 3:
                assert rsolve_hyper(c, q, n).subs(zip(symbols('C:3'), [0, 0, 0])).expand() == p