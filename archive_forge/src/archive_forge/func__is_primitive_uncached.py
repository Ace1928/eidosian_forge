from ..drilling import compute_geodesic_info
from ..drilling.geodesic_tube import GeodesicTube
from ..drilling.line import distance_r13_lines
from ..snap.t3mlite import simplex # type: ignore
def _is_primitive_uncached(self):
    pieces = self._get_pieces_covering_geodesic()
    for i, piece0 in enumerate(pieces):
        for j, piece1 in enumerate(pieces):
            if i < j:
                if piece0.tet == piece1.tet:
                    if _are_parallel_light_vectors(piece0.lifted_geodesic.points[0], piece1.lifted_geodesic.points[0], 1e-05):
                        return False
    return True