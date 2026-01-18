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
def test_type_of_triangle():
    p1 = Polygon(Point(0, 0), Point(5, 0), Point(2, 4))
    assert p1.is_isosceles() == True
    assert p1.is_scalene() == False
    assert p1.is_equilateral() == False
    p2 = Polygon(Point(0, 0), Point(0, 2), Point(4, 0))
    assert p2.is_isosceles() == False
    assert p2.is_scalene() == True
    assert p2.is_equilateral() == False
    p3 = Polygon(Point(0, 0), Point(6, 0), Point(3, sqrt(27)))
    assert p3.is_isosceles() == True
    assert p3.is_scalene() == False
    assert p3.is_equilateral() == True