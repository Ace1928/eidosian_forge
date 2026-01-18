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
def test_roots_cyclotomic():
    assert roots_cyclotomic(cyclotomic_poly(1, x, polys=True)) == [1]
    assert roots_cyclotomic(cyclotomic_poly(2, x, polys=True)) == [-1]
    assert roots_cyclotomic(cyclotomic_poly(3, x, polys=True)) == [Rational(-1, 2) - I * sqrt(3) / 2, Rational(-1, 2) + I * sqrt(3) / 2]
    assert roots_cyclotomic(cyclotomic_poly(4, x, polys=True)) == [-I, I]
    assert roots_cyclotomic(cyclotomic_poly(6, x, polys=True)) == [S.Half - I * sqrt(3) / 2, S.Half + I * sqrt(3) / 2]
    assert roots_cyclotomic(cyclotomic_poly(7, x, polys=True)) == [-cos(pi / 7) - I * sin(pi / 7), -cos(pi / 7) + I * sin(pi / 7), -cos(pi * Rational(3, 7)) - I * sin(pi * Rational(3, 7)), -cos(pi * Rational(3, 7)) + I * sin(pi * Rational(3, 7)), cos(pi * Rational(2, 7)) - I * sin(pi * Rational(2, 7)), cos(pi * Rational(2, 7)) + I * sin(pi * Rational(2, 7))]
    assert roots_cyclotomic(cyclotomic_poly(8, x, polys=True)) == [-sqrt(2) / 2 - I * sqrt(2) / 2, -sqrt(2) / 2 + I * sqrt(2) / 2, sqrt(2) / 2 - I * sqrt(2) / 2, sqrt(2) / 2 + I * sqrt(2) / 2]
    assert roots_cyclotomic(cyclotomic_poly(12, x, polys=True)) == [-sqrt(3) / 2 - I / 2, -sqrt(3) / 2 + I / 2, sqrt(3) / 2 - I / 2, sqrt(3) / 2 + I / 2]
    assert roots_cyclotomic(cyclotomic_poly(1, x, polys=True), factor=True) == [1]
    assert roots_cyclotomic(cyclotomic_poly(2, x, polys=True), factor=True) == [-1]
    assert roots_cyclotomic(cyclotomic_poly(3, x, polys=True), factor=True) == [-root(-1, 3), -1 + root(-1, 3)]
    assert roots_cyclotomic(cyclotomic_poly(4, x, polys=True), factor=True) == [-I, I]
    assert roots_cyclotomic(cyclotomic_poly(5, x, polys=True), factor=True) == [-root(-1, 5), -root(-1, 5) ** 3, root(-1, 5) ** 2, -1 - root(-1, 5) ** 2 + root(-1, 5) + root(-1, 5) ** 3]
    assert roots_cyclotomic(cyclotomic_poly(6, x, polys=True), factor=True) == [1 - root(-1, 3), root(-1, 3)]