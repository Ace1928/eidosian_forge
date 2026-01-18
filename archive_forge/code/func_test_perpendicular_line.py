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
def test_perpendicular_line():
    p1, p2, p3 = (Point(0, 0, 0), Point(2, 3, 4), Point(-2, 2, 0))
    l = Line(p1, p2)
    p = l.perpendicular_line(p3)
    assert p.p1 == p3
    assert p.p2 in l
    p1, p2, p3 = (Point(0, 0), Point(2, 3), Point(-2, 2))
    l = Line(p1, p2)
    p = l.perpendicular_line(p3)
    assert p.p1 == p3
    assert p.direction.unit == (p3 - l.projection(p3)).unit