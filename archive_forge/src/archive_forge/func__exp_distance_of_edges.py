from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
@staticmethod
def _exp_distance_of_edges(edges):
    """
        Given edges between two (not necessarily distinct) cusps,
        compute the exp of the smallest (hyperbolic) distance of the
        two cusp neighborhoods measured along all the given edges.
        """
    return correct_min([ComplexCuspCrossSection._exp_distance_edge(edge) for edge in edges])