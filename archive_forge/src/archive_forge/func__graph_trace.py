from .line import R13LineWithMatrix
from . import epsilons
from . import constants
from . import exceptions
from ..hyperboloid import r13_dot, o13_inverse, distance_unit_time_r13_points # type: ignore
from ..snap.t3mlite import simplex # type: ignore
from ..snap.t3mlite import Tetrahedron, Vertex, Mcomplex # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
from ..matrix import matrix # type: ignore
from typing import Tuple, Sequence, Optional, Any
def _graph_trace(self, tet: Tetrahedron) -> Tuple[Tetrahedron, Sequence[int], Optional[int]]:
    """
        Walk from tetrahedron to tetrahedron (transforming start point and
        the other data) to capture the start point in a tetrahedron.
        """
    if self.mcomplex.verified:
        epsilon = 0
        key = _graph_trace_key_verified
    else:
        epsilon = epsilons.compute_epsilon(self.mcomplex.RF)
        key = _graph_trace_key
    entry_cell = simplex.T
    for i in range(constants.graph_trace_max_steps):
        v = self._find_cusp_if_core_curve(tet, entry_cell, epsilon)
        faces_and_signed_distances = [(face, r13_dot(self.unnormalised_start_point, tet.R13_planes[face])) for face in simplex.TwoSubsimplices]
        if v or not any((signed_distance > epsilon for face, signed_distance in faces_and_signed_distances)):
            return (tet, [face for face, signed_distance in faces_and_signed_distances if not signed_distance < -epsilon], v)
        face, worst_distance = max([face_and_signed_distance for face_and_signed_distance in faces_and_signed_distances if face_and_signed_distance[0] != entry_cell], key=key)
        self._transform(tet.O13_matrices[face])
        entry_cell = tet.Gluing[face].image(face)
        tet = tet.Neighbor[face]
    raise exceptions.UnfinishedGraphTraceGeodesicError(constants.graph_trace_max_steps)