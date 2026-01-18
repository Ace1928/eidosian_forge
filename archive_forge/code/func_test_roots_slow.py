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
def test_roots_slow():
    """Just test that calculating these roots does not hang. """
    a, b, c, d, x = symbols('a,b,c,d,x')
    f1 = x ** 2 * c + a / b + x * c * d - a
    f2 = x ** 2 * (a + b * (c - d) * a) + x * a * b * c / (b * d - d) + (a * d - c / d)
    assert list(roots(f1, x).values()) == [1, 1]
    assert list(roots(f2, x).values()) == [1, 1]
    zz, yy, xx, zy, zx, yx, k = symbols('zz,yy,xx,zy,zx,yx,k')
    e1 = (zz - k) * (yy - k) * (xx - k) + zy * yx * zx + zx - zy - yx
    e2 = (zz - k) * yx * yx + zx * (yy - k) * zx + zy * zy * (xx - k)
    assert list(roots(e1 - e2, k).values()) == [1, 1, 1]
    f = x ** 3 + 2 * x ** 2 + 8
    R = list(roots(f).keys())
    assert not any((i for i in [f.subs(x, ri).n(chop=True) for ri in R]))