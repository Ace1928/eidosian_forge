from sympy.core.function import expand
from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.sets.sets import Interval
from sympy.simplify.simplify import simplify
from sympy.physics.continuum_mechanics.beam import Beam
from sympy.functions import SingularityFunction, Piecewise, meijerg, Abs, log
from sympy.testing.pytest import raises
from sympy.physics.units import meter, newton, kilo, giga, milli
from sympy.physics.continuum_mechanics.beam import Beam3D
from sympy.geometry import Circle, Polygon, Point2D, Triangle
from sympy.core.sympify import sympify
def test_point_cflexure():
    E = Symbol('E')
    I = Symbol('I')
    b = Beam(10, E, I)
    b.apply_load(-4, 0, -1)
    b.apply_load(-46, 6, -1)
    b.apply_load(10, 2, -1)
    b.apply_load(20, 4, -1)
    b.apply_load(3, 6, 0)
    assert b.point_cflexure() == [Rational(10, 3)]