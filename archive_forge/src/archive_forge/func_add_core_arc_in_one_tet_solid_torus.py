from ..snap.t3mlite.simplex import *
from ..snap.t3mlite.edge import Edge
from ..snap.t3mlite.arrow import Arrow
from ..snap.t3mlite.tetrahedron import Tetrahedron
from ..snap.t3mlite.mcomplex import VERBOSE
from .exceptions import GeneralPositionError
from .rational_linear_algebra import Vector3, QQ
from . import pl_utils
from . import stored_moves
from .mcomplex_with_expansion import McomplexWithExpansion
from .mcomplex_with_memory import McomplexWithMemory
from .barycentric_geometry import (BarycentricPoint, BarycentricArc,
import random
import collections
import time
def add_core_arc_in_one_tet_solid_torus(mcomplex, tet):
    M = mcomplex
    assert tet.Neighbor[F2] == tet.Neighbor[F3] == tet
    assert no_fixed_point(tet.Gluing[F2]) and no_fixed_point(tet.Gluing[F3])
    c0, c1, c2, c3 = [QQ(x) for x in ['1/3', '1/3', '0', '1/3']]
    p1 = BarycentricPoint(c0, c1, c2, c3)
    p2 = p1.permute(tet.Gluing[F2])
    tet.arcs = [BarycentricArc(p1, p2)]
    M.connect_arcs()