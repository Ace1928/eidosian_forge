from sympy.core import S
from sympy.core.sorting import ordered
from sympy.core.symbol import _symbol, symbols
from sympy.geometry.entity import GeometryEntity, GeometrySet
from sympy.geometry.point import Point, Point2D
from sympy.geometry.line import Line, Line2D, Ray2D, Segment2D, LinearEntity3D
from sympy.geometry.ellipse import Ellipse
from sympy.functions import sign
from sympy.simplify import simplify
from sympy.solvers.solvers import solve
@property
def vertex(self):
    """The vertex of the parabola.

        Returns
        =======

        vertex : Point

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))
        >>> p1.vertex
        Point2D(0, 4)

        """
    focus = self.focus
    m = self.directrix.slope
    if m is S.Infinity:
        vertex = Point(focus.args[0] - self.p_parameter, focus.args[1])
    elif m == 0:
        vertex = Point(focus.args[0], focus.args[1] - self.p_parameter)
    else:
        vertex = self.axis_of_symmetry.intersection(self)[0]
    return vertex