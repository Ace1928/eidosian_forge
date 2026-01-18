from sympy.core.numbers import (I, Integer, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.polyerrors import UnsolvableFactorError
from sympy.polys.polyoptions import Options
from sympy.polys.polytools import Poly
from sympy.solvers.solvers import solve
from sympy.utilities.iterables import flatten
from sympy.abc import x, y, z
from sympy.polys import PolynomialError
from sympy.solvers.polysys import (solve_poly_system,
from sympy.polys.polytools import parallel_poly_from_expr
from sympy.testing.pytest import raises
def test_solve_issue_3686():
    roots = solve_poly_system([(x - 5) ** 2 / 250000 + (y - Rational(5, 10)) ** 2 / 250000 - 1, x], x, y)
    assert roots == [(0, S.Half - 15 * sqrt(1111)), (0, S.Half + 15 * sqrt(1111))]
    roots = solve_poly_system([(x - 5) ** 2 / 250000 + (y - 5.0 / 10) ** 2 / 250000 - 1, x], x, y)
    assert len(roots) == 2
    assert roots[0][0] == 0
    assert roots[0][1].epsilon_eq(-499.474999374969, 1000000000000.0)
    assert roots[1][0] == 0
    assert roots[1][1].epsilon_eq(500.474999374969, 1000000000000.0)