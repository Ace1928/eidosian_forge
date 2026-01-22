from ..snap.t3mlite import Mcomplex
from ..snap.t3mlite import simplex, Tetrahedron
from collections import deque
from typing import Dict
def _walk_tet_to_face(start_tet: Tetrahedron, tet_to_face: Dict[Tetrahedron, int]) -> None:
    tet = start_tet
    while True:
        tet = _walk_face(tet, 1, tet_to_face[tet])
        if tet is start_tet:
            break