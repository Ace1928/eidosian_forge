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
def _find_cusp_if_core_curve(self, tet: Tetrahedron, entry_cell: int, epsilon) -> Optional[int]:
    """
        Check that the lift of the geodesic is close to the lifts of the core
        curves at the vertices of the tetrahedron adjacent to entry_cell
        where entry_cell is either in simplex.TwoSubsimplices or simplex.T.

        If close, returns the vertex of the tetrahedron (in
        simplex.ZeroSubsimplices), else None.
        """
    if not self.line:
        return None
    for v in simplex.ZeroSubsimplices:
        if not simplex.is_subset(v, entry_cell):
            continue
        core_curve = tet.core_curves.get(v)
        if not core_curve:
            continue
        p = [[r13_dot(pt0, pt1) for pt0 in self.line.r13_line.points] for pt1 in tet.core_curves[v].r13_line.points]
        if not (abs(p[0][0]) > epsilon or abs(p[1][1]) > epsilon):
            return v
        if not (abs(p[0][1]) > epsilon or abs(p[1][0]) > epsilon):
            return v
    return None