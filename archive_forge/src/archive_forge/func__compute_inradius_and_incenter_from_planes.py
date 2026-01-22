from .line import R13LineWithMatrix
from ..verify.shapes import compute_hyperbolic_shapes # type: ignore
from ..snap.fundamental_polyhedron import FundamentalPolyhedronEngine # type: ignore
from ..snap.kernel_structures import TransferKernelStructuresEngine # type: ignore
from ..snap.t3mlite import simplex, Mcomplex, Tetrahedron, Vertex # type: ignore
from ..SnapPy import word_as_list # type: ignore
from ..hyperboloid import (o13_inverse,  # type: ignore
from ..upper_halfspace import sl2c_inverse, psl2c_to_o13 # type: ignore
from ..upper_halfspace.ideal_point import ideal_point_to_r13 # type: ignore
from ..matrix import vector, matrix, mat_solve # type: ignore
from ..math_basics import prod, xgcd # type: ignore
from collections import deque
from typing import Tuple, Sequence, Optional, Any
def _compute_inradius_and_incenter_from_planes(planes) -> Tuple[Any, Any]:
    """
    Given outside-facing normals for the four faces of a
    tetrahedron, compute the hyperbolic inradius and the
    incenter (as unit time vector) of the tetrahedron (in the
    hyperboloid model).
    """
    RF = planes[0][0].parent()
    m = matrix([[-plane[0], plane[1], plane[2], plane[3]] for plane in planes])
    v = vector([RF(-1), RF(-1), RF(-1), RF(-1)])
    pt = mat_solve(m, v)
    scale = 1 / (-r13_dot(pt, pt)).sqrt()
    return (scale.arcsinh(), scale * pt)