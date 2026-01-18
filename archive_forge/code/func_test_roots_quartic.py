from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.elementary.complexes import (conjugate, im, re)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, cos, sin)
from sympy.polys.domains.integerring import ZZ
from sympy.sets.sets import Interval
from sympy.simplify.powsimp import powsimp
from sympy.polys import Poly, cyclotomic_poly, intervals, nroots, rootof
from sympy.polys.polyroots import (root_factors, roots_linear,
from sympy.polys.orthopolys import legendre_poly
from sympy.polys.polyerrors import PolynomialError, \
from sympy.polys.polyutils import _nsort
from sympy.testing.pytest import raises, slow
from sympy.core.random import verify_numerically
import mpmath
from itertools import product
def test_roots_quartic():
    assert roots_quartic(Poly(x ** 4, x)) == [0, 0, 0, 0]
    assert roots_quartic(Poly(x ** 4 + x ** 3, x)) in [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]
    assert roots_quartic(Poly(x ** 4 - x ** 3, x)) in [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    lhs = roots_quartic(Poly(x ** 4 + x, x))
    rhs = [S.Half + I * sqrt(3) / 2, S.Half - I * sqrt(3) / 2, S.Zero, -S.One]
    assert sorted(lhs, key=hash) == sorted(rhs, key=hash)
    for i, (a, b, c, d) in enumerate([(1, 2, 3, 0), (3, -7, -9, 9), (1, 2, 3, 4), (1, 2, 3, 4), (-7, -3, 3, -6), (-3, 5, -6, -4), (6, -5, -10, -3)]):
        if i == 2:
            c = -a * (a ** 2 / S(8) - b / S(2))
        elif i == 3:
            d = a * (a * (a ** 2 * Rational(3, 256) - b / S(16)) + c / S(4))
        eq = x ** 4 + a * x ** 3 + b * x ** 2 + c * x + d
        ans = roots_quartic(Poly(eq, x))
        assert all((eq.subs(x, ai).n(chop=True) == 0 for ai in ans))
    eq = Poly(q * x + q / 4 + x ** 4 + x ** 3 + 2 * x ** 2 - Rational(1, 3), x)
    sol = roots_quartic(eq)
    assert all((verify_numerically(eq.subs(x, i), 0) for i in sol))
    z = symbols('z', negative=True)
    eq = x ** 4 + 2 * x ** 3 + 3 * x ** 2 + x * (z + 11) + 5
    zans = roots_quartic(Poly(eq, x))
    assert all([verify_numerically(eq.subs(((x, i), (z, -1))), 0) for i in zans])
    eq = Poly(y * x ** 4 + x ** 3 - x + z, x)
    ans = roots_quartic(eq)
    assert all((type(i) == Piecewise for i in ans))
    reps = ({'y': Rational(-1, 3), 'z': Rational(-1, 4)}, {'y': Rational(-1, 3), 'z': Rational(-1, 2)}, {'y': Rational(-1, 3), 'z': -2})
    for rep in reps:
        sol = roots_quartic(Poly(eq.subs(rep), x))
        assert all([verify_numerically(w.subs(rep) - s, 0) for w, s in zip(ans, sol)])