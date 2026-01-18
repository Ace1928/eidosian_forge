from functools import cmp_to_key
from sympy.abc import x, y, z
from sympy.core import S, diff, Expr, Symbol
from sympy.core.sympify import _sympify
from sympy.geometry import Segment2D, Polygon, Point, Point2D
from sympy.polys.polytools import LC, gcd_list, degree_list, Poly
from sympy.simplify.simplify import nsimplify
def x_axis_cut(ls):
    """Returns the point where the input line segment
        intersects the x-axis.

        Parameters
        ==========

        ls :
            Line segment
        """
    p, q = ls.points
    if p.y.is_zero:
        return tuple(p)
    elif q.y.is_zero:
        return tuple(q)
    elif p.y / q.y < S.Zero:
        return (p.y * (p.x - q.x) / (q.y - p.y) + p.x, S.Zero)
    else:
        return ()