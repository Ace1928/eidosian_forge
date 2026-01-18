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
def test_root_factors():
    assert root_factors(Poly(1, x)) == [Poly(1, x)]
    assert root_factors(Poly(x, x)) == [Poly(x, x)]
    assert root_factors(x ** 2 - 1, x) == [x + 1, x - 1]
    assert root_factors(x ** 2 - y, x) == [x - sqrt(y), x + sqrt(y)]
    assert root_factors((x ** 4 - 1) ** 2) == [x + 1, x + 1, x - 1, x - 1, x - I, x - I, x + I, x + I]
    assert root_factors(Poly(x ** 4 - 1, x), filter='Z') == [Poly(x + 1, x), Poly(x - 1, x), Poly(x ** 2 + 1, x)]
    assert root_factors(8 * x ** 2 + 12 * x ** 4 + 6 * x ** 6 + x ** 8, x, filter='Q') == [x, x, x ** 6 + 6 * x ** 4 + 12 * x ** 2 + 8]