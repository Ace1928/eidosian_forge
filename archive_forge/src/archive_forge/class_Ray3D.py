from sympy.core.containers import Tuple
from sympy.core.evalf import N
from sympy.core.expr import Expr
from sympy.core.numbers import Rational, oo, Float
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import _symbol, Dummy, uniquely_named_symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (_pi_coeff, acos, tan, atan2)
from .entity import GeometryEntity, GeometrySet
from .exceptions import GeometryError
from .point import Point, Point3D
from .util import find, intersection
from sympy.logic.boolalg import And
from sympy.matrices import Matrix
from sympy.sets.sets import Intersection
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve
from sympy.solvers.solveset import linear_coeffs
from sympy.utilities.misc import Undecidable, filldedent
import random
class Ray3D(LinearEntity3D, Ray):
    """
    A Ray is a semi-line in the space with a source point and a direction.

    Parameters
    ==========

    p1 : Point3D
        The source of the Ray
    p2 : Point or a direction vector
    direction_ratio: Determines the direction in which the Ray propagates.


    Attributes
    ==========

    source
    xdirection
    ydirection
    zdirection

    See Also
    ========

    sympy.geometry.point.Point3D, Line3D


    Examples
    ========

    >>> from sympy import Point3D, Ray3D
    >>> r = Ray3D(Point3D(2, 3, 4), Point3D(3, 5, 0))
    >>> r
    Ray3D(Point3D(2, 3, 4), Point3D(3, 5, 0))
    >>> r.points
    (Point3D(2, 3, 4), Point3D(3, 5, 0))
    >>> r.source
    Point3D(2, 3, 4)
    >>> r.xdirection
    oo
    >>> r.ydirection
    oo
    >>> r.direction_ratio
    [1, 2, -4]

    """

    def __new__(cls, p1, pt=None, direction_ratio=(), **kwargs):
        if isinstance(p1, LinearEntity3D):
            if pt is not None:
                raise ValueError('If p1 is a LinearEntity, pt must be None')
            p1, pt = p1.args
        else:
            p1 = Point(p1, dim=3)
        if pt is not None and len(direction_ratio) == 0:
            pt = Point(pt, dim=3)
        elif len(direction_ratio) == 3 and pt is None:
            pt = Point3D(p1.x + direction_ratio[0], p1.y + direction_ratio[1], p1.z + direction_ratio[2])
        else:
            raise ValueError(filldedent('\n                A 2nd Point or keyword "direction_ratio" must be used.\n            '))
        return LinearEntity3D.__new__(cls, p1, pt, **kwargs)

    @property
    def xdirection(self):
        """The x direction of the ray.

        Positive infinity if the ray points in the positive x direction,
        negative infinity if the ray points in the negative x direction,
        or 0 if the ray is vertical.

        See Also
        ========

        ydirection

        Examples
        ========

        >>> from sympy import Point3D, Ray3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(0, -1, 0)
        >>> r1, r2 = Ray3D(p1, p2), Ray3D(p1, p3)
        >>> r1.xdirection
        oo
        >>> r2.xdirection
        0

        """
        if self.p1.x < self.p2.x:
            return S.Infinity
        elif self.p1.x == self.p2.x:
            return S.Zero
        else:
            return S.NegativeInfinity

    @property
    def ydirection(self):
        """The y direction of the ray.

        Positive infinity if the ray points in the positive y direction,
        negative infinity if the ray points in the negative y direction,
        or 0 if the ray is horizontal.

        See Also
        ========

        xdirection

        Examples
        ========

        >>> from sympy import Point3D, Ray3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(-1, -1, -1), Point3D(-1, 0, 0)
        >>> r1, r2 = Ray3D(p1, p2), Ray3D(p1, p3)
        >>> r1.ydirection
        -oo
        >>> r2.ydirection
        0

        """
        if self.p1.y < self.p2.y:
            return S.Infinity
        elif self.p1.y == self.p2.y:
            return S.Zero
        else:
            return S.NegativeInfinity

    @property
    def zdirection(self):
        """The z direction of the ray.

        Positive infinity if the ray points in the positive z direction,
        negative infinity if the ray points in the negative z direction,
        or 0 if the ray is horizontal.

        See Also
        ========

        xdirection

        Examples
        ========

        >>> from sympy import Point3D, Ray3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(-1, -1, -1), Point3D(-1, 0, 0)
        >>> r1, r2 = Ray3D(p1, p2), Ray3D(p1, p3)
        >>> r1.ydirection
        -oo
        >>> r2.ydirection
        0
        >>> r2.zdirection
        0

        """
        if self.p1.z < self.p2.z:
            return S.Infinity
        elif self.p1.z == self.p2.z:
            return S.Zero
        else:
            return S.NegativeInfinity