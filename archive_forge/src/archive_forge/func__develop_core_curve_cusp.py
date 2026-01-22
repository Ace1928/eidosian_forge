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
def _develop_core_curve_cusp(mcomplex: Mcomplex, v: Vertex, core_curve: R13LineWithMatrix) -> None:
    """
    Given the core curve computed from the SnapPea kernel's given
    words for the meridian and longitude for the given cusp,
    compute the lift of the core curve for all vertices of the
    tetrahedra corresponding to the given cusp.
    """
    tet, vertex = _find_standard_basepoint(mcomplex, v)
    tet.core_curves[vertex] = core_curve
    pending_tet_verts = deque([(tet, vertex, core_curve)])
    while pending_tet_verts:
        tet, vertex, core_curve = pending_tet_verts.popleft()
        for f in simplex.FacesAroundVertexCounterclockwise[vertex]:
            new_tet = tet.Neighbor[f]
            new_vertex = tet.Gluing[f].image(vertex)
            if new_vertex in new_tet.core_curves:
                continue
            new_core_curve = core_curve.transformed(tet.O13_matrices[f])
            new_tet.core_curves[new_vertex] = new_core_curve
            pending_tet_verts.append((new_tet, new_vertex, new_core_curve))