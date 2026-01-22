from ..drilling import compute_geodesic_info
from ..drilling.geodesic_tube import GeodesicTube
from ..drilling.line import distance_r13_lines
from ..snap.t3mlite import simplex # type: ignore
class GeodesicTubeInfo:

    def __init__(self, mcomplex, word, index, is_primitive=None):
        self.geodesic_info = compute_geodesic_info(mcomplex, word)
        if not self.geodesic_info.core_curve_cusp:
            self.geodesic_tube = GeodesicTube(mcomplex, self.geodesic_info)
        t = self.geodesic_info.trace
        self.complex_length = _normalize_complex_length(2 * (t / 2).arccosh())
        self.words = [word]
        self.index = index
        RF = t.real().parent()
        self.dist_to_core_curve = RF(1e+50)
        self._pieces_covering_geodesic = []
        self._is_primitive = is_primitive

    def compute_tets_and_R13_endpoints_and_radius_for_tube(self, radius):
        while True:
            safe_radius = self.dist_to_core_curve * 0.98
            if radius > safe_radius:
                radius = safe_radius
                break
            if self.geodesic_tube.covered_radius() > radius:
                break
            self.geodesic_tube._add_next_piece()
            piece = self.geodesic_tube.pieces[-1]
            for v in simplex.ZeroSubsimplices:
                core_curve = piece.tet.core_curves.get(v, None)
                if core_curve:
                    d = distance_r13_lines(core_curve.r13_line, piece.lifted_geodesic)
                    if d < self.dist_to_core_curve:
                        self.dist_to_core_curve = d
        result = []
        for piece in self.geodesic_tube.pieces:
            if piece.lower_bound > radius:
                break
            result.append((piece.tet.Index, [piece.tet.to_coordinates_in_symmetric_tet * pt for pt in piece.lifted_geodesic.points]))
        return (result, radius)

    def _get_pieces_covering_geodesic(self):
        if not self._pieces_covering_geodesic:
            self.geodesic_tube.add_pieces_for_radius(0)
            for piece in self.geodesic_tube.pieces:
                if piece.lower_bound > 0:
                    break
                self._pieces_covering_geodesic.append(piece)
        return self._pieces_covering_geodesic

    def __eq__(self, other):
        diff = _normalize_complex_length(self.complex_length - other.complex_length)
        if not abs(diff) < 0.001:
            return False
        self_cusp = self.geodesic_info.core_curve_cusp
        other_cusp = other.geodesic_info.core_curve_cusp
        if self_cusp or other_cusp:
            if self_cusp and other_cusp:
                return self_cusp.Index == other_cusp.Index
            return False
        piece = self._get_pieces_covering_geodesic()[0]
        point = piece.lifted_geodesic.points[0]
        for other_piece in other._get_pieces_covering_geodesic():
            if piece.tet == other_piece.tet:
                for other_point in other_piece.lifted_geodesic.points:
                    if _are_parallel_light_vectors(point, other_point, 1e-05):
                        return True
        return False

    def is_primitive(self):
        if self._is_primitive is None:
            self._is_primitive = self._is_primitive_uncached()
        return self._is_primitive

    def _is_primitive_uncached(self):
        pieces = self._get_pieces_covering_geodesic()
        for i, piece0 in enumerate(pieces):
            for j, piece1 in enumerate(pieces):
                if i < j:
                    if piece0.tet == piece1.tet:
                        if _are_parallel_light_vectors(piece0.lifted_geodesic.points[0], piece1.lifted_geodesic.points[0], 1e-05):
                            return False
        return True