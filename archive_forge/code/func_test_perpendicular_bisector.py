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
def test_perpendicular_bisector():
    s1 = Segment(Point(0, 0), Point(1, 1))
    aline = Line(Point(S.Half, S.Half), Point(Rational(3, 2), Rational(-1, 2)))
    on_line = Segment(Point(S.Half, S.Half), Point(Rational(3, 2), Rational(-1, 2))).midpoint
    assert s1.perpendicular_bisector().equals(aline)
    assert s1.perpendicular_bisector(on_line).equals(Segment(s1.midpoint, on_line))
    assert s1.perpendicular_bisector(on_line + (1, 0)).equals(aline)