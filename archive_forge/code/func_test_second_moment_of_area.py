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
def test_second_moment_of_area():
    x, y = symbols('x, y')
    p1, p2, p3 = [(0, 0), (4, 0), (0, 2)]
    p = (0, 0)
    eq_y = (1 - x / 4) * 2
    I_yy = integrate(x ** 2 * integrate(1, (y, 0, eq_y)), (x, 0, 4))
    I_xx = integrate(1 * integrate(y ** 2, (y, 0, eq_y)), (x, 0, 4))
    I_xy = integrate(x * integrate(y, (y, 0, eq_y)), (x, 0, 4))
    triangle = Polygon(p1, p2, p3)
    assert I_xx - triangle.second_moment_of_area(p)[0] == 0
    assert I_yy - triangle.second_moment_of_area(p)[1] == 0
    assert I_xy - triangle.second_moment_of_area(p)[2] == 0
    p1, p2, p3, p4 = [(0, 0), (4, 0), (4, 2), (0, 2)]
    I_yy = integrate(x ** 2 * integrate(1, (y, 0, 2)), (x, 0, 4))
    I_xx = integrate(1 * integrate(y ** 2, (y, 0, 2)), (x, 0, 4))
    I_xy = integrate(x * integrate(y, (y, 0, 2)), (x, 0, 4))
    rectangle = Polygon(p1, p2, p3, p4)
    assert I_xx - rectangle.second_moment_of_area(p)[0] == 0
    assert I_yy - rectangle.second_moment_of_area(p)[1] == 0
    assert I_xy - rectangle.second_moment_of_area(p)[2] == 0
    r = RegularPolygon(Point(0, 0), 5, 3)
    assert r.second_moment_of_area() == (1875 * sqrt(3) / S(32), 1875 * sqrt(3) / S(32), 0)