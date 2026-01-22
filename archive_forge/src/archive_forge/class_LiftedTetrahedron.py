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
class LiftedTetrahedron:
    """
    Represents the lift of a tetrahedron in a manifold to the hyperboloid
    model.

    That is, if a tetrahedron (as part of the fundamental domain) was assigned
    vertices by calling add_r13_geometry, then the vertices of a
    LiftedTetrahedron l will be given by l.o13_matrices * tet.R13_vertices[v]
    where v in snappy.snap.t3mlite.simplex.ZeroSubsimplices.
    """

    def __init__(self, tet: Tetrahedron, o13_matrix):
        self.tet = tet
        self.o13_matrix = o13_matrix