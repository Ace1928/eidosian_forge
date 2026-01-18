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
def three_to_two_arc_transfer(old_arrow, new_arrows, north_pole=None):
    a = old_arrow
    b_orig, b, c = new_arrows
    arcs_in_R3 = []
    embeds = barycentric_edge_embedding(a.glued().glued(), north_pole)
    for old_tet, emb in [embeds[1], embeds[2], embeds[0]]:
        arcs_in_R3.extend(emb.transfer_arcs_to_R3(old_tet.arcs))
    glue_up_arcs_in_R3(arcs_in_R3)
    straighten_arcs(arcs_in_R3)
    for new_tet, emb in barycentric_face_embedding(b_orig, north_pole):
        new_tet.arcs = arcs_to_add(emb.transfer_arcs_from_R3(arcs_in_R3))