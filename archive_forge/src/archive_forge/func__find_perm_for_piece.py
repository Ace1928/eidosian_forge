from .cusps import CuspPostDrillInfo
from .tracing import GeodesicPiece
from .peripheral_curves import install_peripheral_curves
from ..snap.t3mlite import Tetrahedron, Perm4, Mcomplex, simplex
from typing import Dict, Tuple, List, Sequence
def _find_perm_for_piece(piece: GeodesicPiece):
    """
    Given a GeodesicPiece with endpoints being on the vertices
    of the tetrahedron and spanning an oriented edge of the tetrahedron,
    find an "edge embedding permutation" (similar to regina's
    edge embedding) that maps the 0-1 edge to the given edge.

    The subtetrahedron corresponding to this permutation is
    adjacent to half of this edge.
    """
    s0 = piece.endpoints[0].subsimplex
    s1 = piece.endpoints[1].subsimplex
    for perm in Perm4.A4():
        if perm.image(simplex.V0) == s0 and perm.image(simplex.V1) == s1:
            return perm