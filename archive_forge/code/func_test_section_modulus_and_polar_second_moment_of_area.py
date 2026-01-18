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
def test_section_modulus_and_polar_second_moment_of_area():
    a, b = symbols('a, b', positive=True)
    x, y = symbols('x, y')
    rectangle = Polygon((0, b), (0, 0), (a, 0), (a, b))
    assert rectangle.section_modulus(Point(x, y)) == (a * b ** 3 / 12 / (-b / 2 + y), a ** 3 * b / 12 / (-a / 2 + x))
    assert rectangle.polar_second_moment_of_area() == a ** 3 * b / 12 + a * b ** 3 / 12
    convex = RegularPolygon((0, 0), 1, 6)
    assert convex.section_modulus() == (Rational(5, 8), sqrt(3) * Rational(5, 16))
    assert convex.polar_second_moment_of_area() == 5 * sqrt(3) / S(8)
    concave = Polygon((0, 0), (1, 8), (3, 4), (4, 6), (7, 1))
    assert concave.section_modulus() == (Rational(-6371, 429), Rational(-9778, 519))
    assert concave.polar_second_moment_of_area() == Rational(-38669, 252)