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
class _PendingPiece:
    """
    A lifted tetrahedron that still needs to be processed by GeodesicTube
    together with the face through which this lifted tetrahedron was
    reached.

    The lifted tetrahedron lives in the quotient space of the hyperboloid
    model by (powers of) the matrix corresponding to the closed geodesic,
    see ZQuotientLiftedTetrahedronSet.

    The algorithm in GeodesicTube might add the same lifted tetrahedron
    multiple times to the queue of pending pieces as there are four
    neighboring lifted tetrahedra from which this lifted tetrahedron can
    be reached.

    Let L be the line (in the quotient space) about which we develop the
    geodesic tube. lower_bound is a lower bound on the distance between
    L and the face through which this lifted tetrahedron was reached.
    Note that lower_bound might be larger than the distance between L and
    this lifted tetrahedron (which is the minimum of all distances between
    L and any of the faces of this lifted tetrahedron).

    The < operator is overloaded so that the piece with the lowest
    lower_bound will be picked up next by a priority queue.

    If pieces are processed in this order, then the lower_bound of the
    next piece will actually be a lower bound for the distance between L
    and the lifted tetrahedron (with other pending pieces for the same
    lifted tetrahedron having higher values for lower_bound and thus
    being further down the queue).
    """

    def __init__(self, lifted_tetrahedron: LiftedTetrahedron, lower_bound, entry_cell: int=simplex.T):
        self.lifted_tetrahedron = lifted_tetrahedron
        self.lower_bound = lower_bound
        self.entry_cell = entry_cell
        if is_RealIntervalFieldElement(lower_bound):
            if lower_bound.is_NaN():
                raise InsufficientPrecisionError('A NaN was encountered while developing a tube about a geodesic. Increasing the precision will probably fix this.')
            self._key = lower_bound.lower()
        else:
            self._key = lower_bound

    def __lt__(self, other):
        return self._key < other._key