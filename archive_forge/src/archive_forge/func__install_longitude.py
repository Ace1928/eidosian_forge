from ..snap.t3mlite import Mcomplex
from ..snap.t3mlite import simplex, Tetrahedron
from collections import deque
from typing import Dict
def _install_longitude(start_tet: Tetrahedron):
    """
    Uses the meridian installed with _install_meridian to
    find a curve crossing the meridian once.
    """
    tet0 = start_tet
    tet1 = start_tet.Neighbor[simplex.F2]
    tet2 = tet1.Neighbor[simplex.F3]
    if not _has_meridian(tet0):
        raise Exception('start_tet expected to have meridian.')
    if not _has_meridian(tet1):
        raise Exception('F2-neighbor of start_tet expected to have meridian.')
    if _has_meridian(tet2):
        raise Exception('F3-enighbor of F2-neighbor of start_tet not expected to have meridian.')
    visited_tet_to_face = {tet1: simplex.F3}
    pending_tets = deque([(tet0, simplex.F2)])
    while True:
        tet, entry_f = pending_tets.popleft()
        if tet in visited_tet_to_face:
            continue
        visited_tet_to_face[tet] = entry_f
        if tet is tet2:
            break
        for f in [simplex.F1, simplex.F2, simplex.F3]:
            neighbor = tet.Neighbor[f]
            if f != entry_f and (not _has_meridian(neighbor)):
                pending_tets.append((neighbor, f))
    _walk_tet_to_face(start_tet, visited_tet_to_face)