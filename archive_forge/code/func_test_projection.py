from sympy.core.numbers import (Float, Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, cos, sin)
from sympy.sets import EmptySet
from sympy.simplify.simplify import simplify
from sympy.functions.elementary.trigonometric import tan
from sympy.geometry import (Circle, GeometryError, Line, Point, Ray,
from sympy.geometry.line import Undecidable
from sympy.geometry.polygon import _asa as asa
from sympy.utilities.iterables import cartes
from sympy.testing.pytest import raises, warns
def test_projection():
    p1 = Point(0, 0)
    p2 = Point3D(0, 0, 0)
    p3 = Point(-x1, x1)
    l1 = Line(p1, Point(1, 1))
    l2 = Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0))
    l3 = Line3D(p2, Point3D(1, 1, 1))
    r1 = Ray(Point(1, 1), Point(2, 2))
    s1 = Segment(Point2D(0, 0), Point2D(0, 1))
    s2 = Segment(Point2D(1, 0), Point2D(2, 1 / 2))
    assert Line(Point(x1, x1), Point(y1, y1)).projection(Point(y1, y1)) == Point(y1, y1)
    assert Line(Point(x1, x1), Point(x1, 1 + x1)).projection(Point(1, 1)) == Point(x1, 1)
    assert Segment(Point(-2, 2), Point(0, 4)).projection(r1) == Segment(Point(-1, 3), Point(0, 4))
    assert Segment(Point(0, 4), Point(-2, 2)).projection(r1) == Segment(Point(0, 4), Point(-1, 3))
    assert s2.projection(s1) == EmptySet
    assert l1.projection(p3) == p1
    assert l1.projection(Ray(p1, Point(-1, 5))) == Ray(Point(0, 0), Point(2, 2))
    assert l1.projection(Ray(p1, Point(-1, 1))) == p1
    assert r1.projection(Ray(Point(1, 1), Point(-1, -1))) == Point(1, 1)
    assert r1.projection(Ray(Point(0, 4), Point(-1, -5))) == Segment(Point(1, 1), Point(2, 2))
    assert r1.projection(Segment(Point(-1, 5), Point(-5, -10))) == Segment(Point(1, 1), Point(2, 2))
    assert r1.projection(Ray(Point(1, 1), Point(-1, -1))) == Point(1, 1)
    assert r1.projection(Ray(Point(0, 4), Point(-1, -5))) == Segment(Point(1, 1), Point(2, 2))
    assert r1.projection(Segment(Point(-1, 5), Point(-5, -10))) == Segment(Point(1, 1), Point(2, 2))
    assert l3.projection(Ray3D(p2, Point3D(-1, 5, 0))) == Ray3D(Point3D(0, 0, 0), Point3D(Rational(4, 3), Rational(4, 3), Rational(4, 3)))
    assert l3.projection(Ray3D(p2, Point3D(-1, 1, 1))) == Ray3D(Point3D(0, 0, 0), Point3D(Rational(1, 3), Rational(1, 3), Rational(1, 3)))
    assert l2.projection(Point3D(5, 5, 0)) == Point3D(5, 0)
    assert l2.projection(Line3D(Point3D(0, 1, 0), Point3D(1, 1, 0))).equals(l2)