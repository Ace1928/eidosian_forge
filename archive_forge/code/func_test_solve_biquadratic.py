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
def test_solve_biquadratic():
    x0, y0, x1, y1, r = symbols('x0 y0 x1 y1 r')
    f_1 = (x - 1) ** 2 + (y - 1) ** 2 - r ** 2
    f_2 = (x - 2) ** 2 + (y - 2) ** 2 - r ** 2
    s = sqrt(2 * r ** 2 - 1)
    a = (3 - s) / 2
    b = (3 + s) / 2
    assert solve_poly_system([f_1, f_2], x, y) == [(a, b), (b, a)]
    f_1 = (x - 1) ** 2 + (y - 2) ** 2 - r ** 2
    f_2 = (x - 1) ** 2 + (y - 1) ** 2 - r ** 2
    assert solve_poly_system([f_1, f_2], x, y) == [(1 - sqrt((2 * r - 1) * (2 * r + 1)) / 2, Rational(3, 2)), (1 + sqrt((2 * r - 1) * (2 * r + 1)) / 2, Rational(3, 2))]
    query = lambda expr: expr.is_Pow and expr.exp is S.Half
    f_1 = (x - 1) ** 2 + (y - 2) ** 2 - r ** 2
    f_2 = (x - x1) ** 2 + (y - 1) ** 2 - r ** 2
    result = solve_poly_system([f_1, f_2], x, y)
    assert len(result) == 2 and all((len(r) == 2 for r in result))
    assert all((r.count(query) == 1 for r in flatten(result)))
    f_1 = (x - x0) ** 2 + (y - y0) ** 2 - r ** 2
    f_2 = (x - x1) ** 2 + (y - y1) ** 2 - r ** 2
    result = solve_poly_system([f_1, f_2], x, y)
    assert len(result) == 2 and all((len(r) == 2 for r in result))
    assert all((len(r.find(query)) == 1 for r in flatten(result)))
    s1 = (x * y - y, x ** 2 - x)
    assert solve(s1) == [{x: 1}, {x: 0, y: 0}]
    s2 = (x * y - x, y ** 2 - y)
    assert solve(s2) == [{y: 1}, {x: 0, y: 0}]
    gens = (x, y)
    for seq in (s1, s2):
        (f, g), opt = parallel_poly_from_expr(seq, *gens)
        raises(SolveFailed, lambda: solve_biquadratic(f, g, opt))
    seq = (x ** 2 + y ** 2 - 2, y ** 2 - 1)
    (f, g), opt = parallel_poly_from_expr(seq, *gens)
    assert solve_biquadratic(f, g, opt) == [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    ans = [(0, -1), (0, 1)]
    seq = (x ** 2 + y ** 2 - 1, y ** 2 - 1)
    (f, g), opt = parallel_poly_from_expr(seq, *gens)
    assert solve_biquadratic(f, g, opt) == ans
    seq = (x ** 2 + y ** 2 - 1, x ** 2 - x + y ** 2 - 1)
    (f, g), opt = parallel_poly_from_expr(seq, *gens)
    assert solve_biquadratic(f, g, opt) == ans