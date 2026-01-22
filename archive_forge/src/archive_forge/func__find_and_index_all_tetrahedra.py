from .moves import one_four_move, two_three_move
from .tracing import GeodesicPiece, GeodesicPieceTracker
from .exceptions import GeodesicStartingPiecesCrossSameFaceError
from . import debug
from ..snap.t3mlite import Mcomplex, Tetrahedron
from typing import Sequence, Dict
def _find_and_index_all_tetrahedra(tet: Tetrahedron):
    """
    Recursively traverses neighbors of the given Tetrahedron
    to find all tetrahedra tet in the connected component.

    Assigns tet.Index to them.
    """
    result = []
    pending_tets = [tet]
    visited_tets = set()
    i = 0
    while pending_tets:
        tet = pending_tets.pop()
        if tet not in visited_tets:
            visited_tets.add(tet)
            tet.Index = i
            i += 1
            result.append(tet)
            for neighbor in tet.Neighbor.values():
                pending_tets.append(neighbor)
    return result