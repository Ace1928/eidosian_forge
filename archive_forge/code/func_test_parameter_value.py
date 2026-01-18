from sympy.core.numbers import (Float, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, cos, sin)
from sympy.functions.elementary.trigonometric import tan
from sympy.geometry import (Circle, Ellipse, GeometryError, Point, Point2D,
from sympy.testing.pytest import raises, slow, warns
from sympy.core.random import verify_numerically
from sympy.geometry.polygon import rad, deg
from sympy.integrals.integrals import integrate
def test_parameter_value():
    t = Symbol('t')
    sq = Polygon((0, 0), (0, 1), (1, 1), (1, 0))
    assert sq.parameter_value((0.5, 1), t) == {t: Rational(3, 8)}
    q = Polygon((0, 0), (2, 1), (2, 4), (4, 0))
    assert q.parameter_value((4, 0), t) == {t: -6 + 3 * sqrt(5)}
    raises(ValueError, lambda: sq.parameter_value((5, 6), t))
    raises(ValueError, lambda: sq.parameter_value(Circle(Point(0, 0), 1), t))