from . import constants
from . import epsilons
from . import exceptions
from .geodesic_tube import add_structures_necessary_for_tube, GeodesicTube
from .geodesic_info import GeodesicInfo
from .line import R13Line, distance_r13_lines
from ..hyperboloid import ( # type: ignore
from ..snap.t3mlite import Mcomplex # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
from ..matrix import vector # type: ignore
from ..math_basics import correct_min # type: ignore
from typing import Sequence, List

    Given a triangulation with structures added by add_r13_geometry
    and GeodesicInfo's with start points on the line that is a lift
    of the closed geodesic, perturbs the start point away from the line
    and computes a new end point (as image of the new start point under
    the matrix associated to the geodesic line). The line segment
    from the start point to the end point forms a simple closed
    curve in the manifold which is guaranteed to be isotopic to the
    closed geodesic.

    If several GeodesicInfo's are given and/or there are filled
    cusps with core curves, the system of simple closed curves
    resulting from the perturbation together with the core curves
    is guaranteed to be isotopic to the original system of closed
    geodesics together with the core curves.

    Through the perturbation, the simple closed curve should avoid
    the 1-skeleton (more precision might be required to see this).
    In particular, the start point should be in the interior of
    a tetrahedron and trace_geodesic should succeed.
    An example where trace_geodesic would not succeed without
    perturbation is e.g., the geodesic 'a' in m125 which lies entirely
    in the 2-skeleton.
    