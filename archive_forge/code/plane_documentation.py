from sympy.core import Dummy, Rational, S, Symbol
from sympy.core.symbol import _symbol
from sympy.functions.elementary.trigonometric import cos, sin, acos, asin, sqrt
from .entity import GeometryEntity
from .line import (Line, Ray, Segment, Line3D, LinearEntity, LinearEntity3D,
from .point import Point, Point3D
from sympy.matrices import Matrix
from sympy.polys.polytools import cancel
from sympy.solvers import solve, linsolve
from sympy.utilities.iterables import uniq, is_sequence
from sympy.utilities.misc import filldedent, func_name, Undecidable
from mpmath.libmp.libmpf import prec_to_dps
import random
Return the parameter(s) corresponding to the given point.

        Examples
        ========

        >>> from sympy import pi, Plane
        >>> from sympy.abc import t, u, v
        >>> p = Plane((2, 0, 0), (0, 0, 1), (0, 1, 0))

        By default, the parameter value returned defines a point
        that is a distance of 1 from the Plane's p1 value and
        in line with the given point:

        >>> on_circle = p.arbitrary_point(t).subs(t, pi/4)
        >>> on_circle.distance(p.p1)
        1
        >>> p.parameter_value(on_circle, t)
        {t: pi/4}

        Moving the point twice as far from p1 does not change
        the parameter value:

        >>> off_circle = p.p1 + (on_circle - p.p1)*2
        >>> off_circle.distance(p.p1)
        2
        >>> p.parameter_value(off_circle, t)
        {t: pi/4}

        If the 2-value parameter is desired, supply the two
        parameter symbols and a replacement dictionary will
        be returned:

        >>> p.parameter_value(on_circle, u, v)
        {u: sqrt(10)/10, v: sqrt(10)/30}
        >>> p.parameter_value(off_circle, u, v)
        {u: sqrt(10)/5, v: sqrt(10)/15}
        