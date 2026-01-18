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
def test_roots_strict():
    assert roots(x ** 2 - 2 * x + 1, strict=False) == {1: 2}
    assert roots(x ** 2 - 2 * x + 1, strict=True) == {1: 2}
    assert roots(x ** 6 - 2 * x ** 5 - x ** 2 + 3 * x - 2, strict=False) == {2: 1}
    raises(UnsolvableFactorError, lambda: roots(x ** 6 - 2 * x ** 5 - x ** 2 + 3 * x - 2, strict=True))