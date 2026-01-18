from functools import singledispatch
from sympy.core.numbers import pi
from sympy.functions.elementary.trigonometric import tan
from sympy.simplify import trigsimp
from sympy.core import Basic, Tuple
from sympy.core.symbol import _symbol
from sympy.solvers import solve
from sympy.geometry import Point, Segment, Curve, Ellipse, Polygon
from sympy.vector import ImplicitRegion

    Returns a list of ParametricRegion objects representing the geometric region.

    Examples
    ========

    >>> from sympy.abc import t
    >>> from sympy.vector import parametric_region_list
    >>> from sympy.geometry import Point, Curve, Ellipse, Segment, Polygon

    >>> p = Point(2, 5)
    >>> parametric_region_list(p)
    [ParametricRegion((2, 5))]

    >>> c = Curve((t**3, 4*t), (t, -3, 4))
    >>> parametric_region_list(c)
    [ParametricRegion((t**3, 4*t), (t, -3, 4))]

    >>> e = Ellipse(Point(1, 3), 2, 3)
    >>> parametric_region_list(e)
    [ParametricRegion((2*cos(t) + 1, 3*sin(t) + 3), (t, 0, 2*pi))]

    >>> s = Segment(Point(1, 3), Point(2, 6))
    >>> parametric_region_list(s)
    [ParametricRegion((t + 1, 3*t + 3), (t, 0, 1))]

    >>> p1, p2, p3, p4 = [(0, 1), (2, -3), (5, 3), (-2, 3)]
    >>> poly = Polygon(p1, p2, p3, p4)
    >>> parametric_region_list(poly)
    [ParametricRegion((2*t, 1 - 4*t), (t, 0, 1)), ParametricRegion((3*t + 2, 6*t - 3), (t, 0, 1)),     ParametricRegion((5 - 7*t, 3), (t, 0, 1)), ParametricRegion((2*t - 2, 3 - 2*t),  (t, 0, 1))]

    