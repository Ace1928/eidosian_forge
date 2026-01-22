from . import constants
from . import exceptions
from . import epsilons
from .line import distance_r13_lines, R13Line, R13LineWithMatrix
from .geodesic_info import GeodesicInfo, LiftedTetrahedron
from .quotient_space import balance_end_points_of_line, ZQuotientLiftedTetrahedronSet
from ..hyperboloid import ( # type: ignore
from ..snap.t3mlite import simplex, Tetrahedron, Mcomplex # type: ignore
from ..matrix import matrix # type: ignore
from ..math_basics import is_RealIntervalFieldElement # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
import heapq
from typing import Sequence, Any
def _add_next_piece(self):
    """
        Finds the pending piece "closest" to the lifted closed geodesic,
        adds it to the result and marks the neighboring lifted tetrahedra
        to the pending queue.

        Here, "closest" is not quite precise because we pick the piece
        with the lowest lower bound for the distance. Also recall that the
        distance of a pending piece is the distance between the lifted
        geodesic L and the entry cell of the lifted tetrahedron, not between
        L and the lifted tetrahedron itself.

        So the right picture to have in mind is: imagine the 2-skeleton
        of the triangulation in the quotient space intersecting the boundary
        of a geodesic tube. As the geodesic tube grows, the intersection
        sweeps through the 2-skeleton. The pending pieces will be processed in
        the order the faces of the 2-skeleton are encountered during the
        sweep.
        """
    while True:
        pending_piece = heapq.heappop(self._pending_pieces)
        if self._visited_lifted_tetrahedra.add(pending_piece.lifted_tetrahedron):
            break
    if self.mcomplex.verified:
        epsilon = 0
    else:
        epsilon = epsilons.compute_tube_injectivity_radius_epsilon(self.mcomplex.RF)
    tet = pending_piece.lifted_tetrahedron.tet
    m = pending_piece.lifted_tetrahedron.o13_matrix
    lifted_geodesic = self._line.transformed(o13_inverse(m))
    for v in simplex.ZeroSubsimplices:
        core_curve = tet.core_curves.get(v, None)
        if core_curve:
            d = distance_r13_lines(core_curve.r13_line, lifted_geodesic)
            if not d > epsilon:
                raise exceptions.GeodesicCloseToCoreCurve()
    self.pieces.append(GeodesicTubePiece(tet=tet, lifted_geodesic=lifted_geodesic, lower_bound=pending_piece.lower_bound))
    for f, new_tet in tet.Neighbor.items():
        if f == pending_piece.entry_cell:
            continue
        entry_face = tet.Gluing[f].image(f)
        heapq.heappush(self._pending_pieces, _PendingPiece(LiftedTetrahedron(new_tet, m * new_tet.O13_matrices[entry_face]), lower_bound_for_distance_line_to_tet_face(lifted_geodesic, tet, f, self.mcomplex.verified), entry_cell=entry_face))