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
def test_roots_inexact():
    R1 = roots(x ** 2 + x + 1, x, multiple=True)
    R2 = roots(x ** 2 + x + 1.0, x, multiple=True)
    for r1, r2 in zip(R1, R2):
        assert abs(r1 - r2) < 1e-12
    f = x ** 4 + 3.0 * sqrt(2.0) * x ** 3 - (78.0 + 24.0 * sqrt(3.0)) * x ** 2 + 144.0 * (2 * sqrt(3.0) + 9.0)
    R1 = roots(f, multiple=True)
    R2 = (-12.7530479110482, -3.85012393732929, 4.89897948556636, 7.46155167569183)
    for r1, r2 in zip(R1, R2):
        assert abs(r1 - r2) < 1e-10