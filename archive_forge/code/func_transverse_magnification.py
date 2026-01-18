from sympy.core.numbers import (Float, I, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan2, cos, sin, tan)
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import cancel
from sympy.series.limits import Limit
from sympy.geometry.line import Ray3D
from sympy.geometry.util import intersection
from sympy.geometry.plane import Plane
from sympy.utilities.iterables import is_sequence
from .medium import Medium
def transverse_magnification(si, so):
    """

    Calculates the transverse magnification, which is the ratio of the
    image size to the object size.

    Parameters
    ==========

    so: sympifiable
        Lens-object distance.

    si: sympifiable
        Lens-image distance.

    Example
    =======

    >>> from sympy.physics.optics import transverse_magnification
    >>> transverse_magnification(30, 15)
    -2

    """
    si = sympify(si)
    so = sympify(so)
    return -(si / so)