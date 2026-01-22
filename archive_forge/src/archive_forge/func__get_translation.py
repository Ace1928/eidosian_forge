from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
@staticmethod
def _get_translation(vertex, ml):
    """
        Compute the translation corresponding to the meridian (ml = 0) or
        longitude (ml = 1) of the given cusp.
        """
    result = 0
    for corner in vertex.Corners:
        tet = corner.Tetrahedron
        subsimplex = corner.Subsimplex
        faces = t3m.simplex.FacesAroundVertexCounterclockwise[subsimplex]
        triangle = tet.horotriangles[subsimplex]
        curves = tet.PeripheralCurves[ml][0][subsimplex]
        for i in range(3):
            this_face = faces[i]
            prev_face = faces[(i + 2) % 3]
            f = curves[this_face] + 2 * curves[prev_face]
            result += f * triangle.lengths[this_face]
    return result / 6