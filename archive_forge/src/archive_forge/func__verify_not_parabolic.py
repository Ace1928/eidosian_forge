from . import exceptions
from . import epsilons
from . import debug
from .tracing import trace_geodesic
from .crush import crush_geodesic_pieces
from .line import R13LineWithMatrix
from .geometric_structure import add_r13_geometry, word_to_psl2c_matrix
from .geodesic_info import GeodesicInfo, sample_line
from .perturb import perturb_geodesics
from .subdivide import traverse_geodesics_to_subdivide
from .cusps import (
from ..snap.t3mlite import Mcomplex
from ..exceptions import InsufficientPrecisionError
import functools
from typing import Sequence
def _verify_not_parabolic(m, mcomplex, word):
    """
    Raise exception when user gives a word corresponding to a parabolic
    matrix.
    """
    if mcomplex.verified:
        epsilon = 0
    else:
        epsilon = epsilons.compute_epsilon(mcomplex.RF)
    tr = m.trace()
    if not (abs(tr - 2) > epsilon and abs(tr + 2) > epsilon):
        raise exceptions.WordAppearsToBeParabolic(word, tr)