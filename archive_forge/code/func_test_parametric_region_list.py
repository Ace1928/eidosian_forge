from sympy.core.numbers import pi
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.parametricregion import ParametricRegion, parametric_region_list
from sympy.geometry import Point, Segment, Curve, Ellipse, Line, Parabola, Polygon
from sympy.testing.pytest import raises
from sympy.abc import a, b, r, t, x, y, z, theta, phi
def test_parametric_region_list():
    point = Point(-5, 12)
    assert parametric_region_list(point) == [ParametricRegion((-5, 12))]
    e = Ellipse(Point(2, 8), 2, 6)
    assert parametric_region_list(e, t) == [ParametricRegion((2 * cos(t) + 2, 6 * sin(t) + 8), (t, 0, 2 * pi))]
    c = Curve((t, t ** 3), (t, 5, 3))
    assert parametric_region_list(c) == [ParametricRegion((t, t ** 3), (t, 5, 3))]
    s = Segment(Point(2, 11, -6), Point(0, 2, 5))
    assert parametric_region_list(s, t) == [ParametricRegion((2 - 2 * t, 11 - 9 * t, 11 * t - 6), (t, 0, 1))]
    s1 = Segment(Point(0, 0), (1, 0))
    assert parametric_region_list(s1, t) == [ParametricRegion((t, 0), (t, 0, 1))]
    s2 = Segment(Point(1, 2, 3), Point(1, 2, 5))
    assert parametric_region_list(s2, t) == [ParametricRegion((1, 2, 2 * t + 3), (t, 0, 1))]
    s3 = Segment(Point(12, 56), Point(12, 56))
    assert parametric_region_list(s3) == [ParametricRegion((12, 56))]
    poly = Polygon((1, 3), (-3, 8), (2, 4))
    assert parametric_region_list(poly, t) == [ParametricRegion((1 - 4 * t, 5 * t + 3), (t, 0, 1)), ParametricRegion((5 * t - 3, 8 - 4 * t), (t, 0, 1)), ParametricRegion((2 - t, 4 - t), (t, 0, 1))]
    p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))
    raises(ValueError, lambda: parametric_region_list(p1))