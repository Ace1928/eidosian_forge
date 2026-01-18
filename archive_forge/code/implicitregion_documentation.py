from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.polytools import gcd
from sympy.sets.sets import Complement
from sympy.core import Basic, Tuple, diff, expand, Eq, Integer
from sympy.core.sorting import ordered
from sympy.core.symbol import _symbol
from sympy.solvers import solveset, nonlinsolve, diophantine
from sympy.polys import total_degree
from sympy.geometry import Point
from sympy.ntheory.factor_ import core

        Returns the rational parametrization of implicit region.

        Examples
        ========

        >>> from sympy import Eq
        >>> from sympy.abc import x, y, z, s, t
        >>> from sympy.vector import ImplicitRegion

        >>> parabola = ImplicitRegion((x, y), y**2 - 4*x)
        >>> parabola.rational_parametrization()
        (4/t**2, 4/t)

        >>> circle = ImplicitRegion((x, y), Eq(x**2 + y**2, 4))
        >>> circle.rational_parametrization()
        (4*t/(t**2 + 1), 4*t**2/(t**2 + 1) - 2)

        >>> I = ImplicitRegion((x, y), x**3 + x**2 - y**2)
        >>> I.rational_parametrization()
        (t**2 - 1, t*(t**2 - 1))

        >>> cubic_curve = ImplicitRegion((x, y), x**3 + x**2 - y**2)
        >>> cubic_curve.rational_parametrization(parameters=(t))
        (t**2 - 1, t*(t**2 - 1))

        >>> sphere = ImplicitRegion((x, y, z), x**2 + y**2 + z**2 - 4)
        >>> sphere.rational_parametrization(parameters=(t, s))
        (-2 + 4/(s**2 + t**2 + 1), 4*s/(s**2 + t**2 + 1), 4*t/(s**2 + t**2 + 1))

        For some conics, regular_points() is unable to find a point on curve.
        To calulcate the parametric representation in such cases, user need
        to determine a point on the region and pass it using reg_point.

        >>> c = ImplicitRegion((x, y), (x  - 1/2)**2 + (y)**2 - (1/4)**2)
        >>> c.rational_parametrization(reg_point=(3/4, 0))
        (0.75 - 0.5/(t**2 + 1), -0.5*t/(t**2 + 1))

        References
        ==========

        - Christoph M. Hoffmann, "Conversion Methods between Parametric and
          Implicit Curves and Surfaces", Purdue e-Pubs, 1990. Available:
          https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=1827&context=cstech

        