from .cusps import CuspPostDrillInfo
from .tracing import GeodesicPiece
from .peripheral_curves import install_peripheral_curves
from ..snap.t3mlite import Tetrahedron, Perm4, Mcomplex, simplex
from typing import Dict, Tuple, List, Sequence
def _traverse_edge(tet0: Tetrahedron, perm0: Perm4, mask: List[bool]):
    """
    Given a subtetrahedron in the barycentric subdivision parametrized
    by a tetrahedron and permutation, find all subtetrahedra adjacent to the
    same edge in the original triangulation. Delete them from the bit mask.
    """
    tet = tet0
    perm = perm0
    while True:
        for p in [perm, perm * _transpositions[0], perm * _transpositions[2], perm * _transpositions[0] * _transpositions[2]]:
            subtet_index = 24 * tet.Index + _perm_to_index(p)
            mask[subtet_index] = False
        face = perm.image(simplex.F3)
        tet, perm = (tet.Neighbor[face], tet.Gluing[face] * perm * Perm4((0, 1, 3, 2)))
        if tet is tet0 and perm.tuple() == perm0.tuple():
            return