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
def test_ray_generation():
    assert Ray((1, 1), angle=pi / 4) == Ray((1, 1), (2, 2))
    assert Ray((1, 1), angle=pi / 2) == Ray((1, 1), (1, 2))
    assert Ray((1, 1), angle=-pi / 2) == Ray((1, 1), (1, 0))
    assert Ray((1, 1), angle=-3 * pi / 2) == Ray((1, 1), (1, 2))
    assert Ray((1, 1), angle=5 * pi / 2) == Ray((1, 1), (1, 2))
    assert Ray((1, 1), angle=5.0 * pi / 2) == Ray((1, 1), (1, 2))
    assert Ray((1, 1), angle=pi) == Ray((1, 1), (0, 1))
    assert Ray((1, 1), angle=3.0 * pi) == Ray((1, 1), (0, 1))
    assert Ray((1, 1), angle=4.0 * pi) == Ray((1, 1), (2, 1))
    assert Ray((1, 1), angle=0) == Ray((1, 1), (2, 1))
    assert Ray((1, 1), angle=4.05 * pi) == Ray(Point(1, 1), Point(2, -sqrt(5) * sqrt(2 * sqrt(5) + 10) / 4 - sqrt(2 * sqrt(5) + 10) / 4 + 2 + sqrt(5)))
    assert Ray((1, 1), angle=4.02 * pi) == Ray(Point(1, 1), Point(2, 1 + tan(4.02 * pi)))
    assert Ray((1, 1), angle=5) == Ray((1, 1), (2, 1 + tan(5)))
    assert Ray3D((1, 1, 1), direction_ratio=[4, 4, 4]) == Ray3D(Point3D(1, 1, 1), Point3D(5, 5, 5))
    assert Ray3D((1, 1, 1), direction_ratio=[1, 2, 3]) == Ray3D(Point3D(1, 1, 1), Point3D(2, 3, 4))
    assert Ray3D((1, 1, 1), direction_ratio=[1, 1, 1]) == Ray3D(Point3D(1, 1, 1), Point3D(2, 2, 2))