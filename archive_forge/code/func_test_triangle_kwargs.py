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
def test_triangle_kwargs():
    assert Triangle(sss=(3, 4, 5)) == Triangle(Point(0, 0), Point(3, 0), Point(3, 4))
    assert Triangle(asa=(30, 2, 30)) == Triangle(Point(0, 0), Point(2, 0), Point(1, sqrt(3) / 3))
    assert Triangle(sas=(1, 45, 2)) == Triangle(Point(0, 0), Point(2, 0), Point(sqrt(2) / 2, sqrt(2) / 2))
    assert Triangle(sss=(1, 2, 5)) is None
    assert deg(rad(180)) == 180