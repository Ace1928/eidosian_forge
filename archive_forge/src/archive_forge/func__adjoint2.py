from ...sage_helper import _within_sage
from .finite_point import *
from .extended_matrix import *
def _adjoint2(m):
    """
    Sage matrix.adjoint() produces an unnecessary large interval for
    ComplexIntervalField entries.
    """
    return matrix([[m[1, 1], -m[0, 1]], [-m[1, 0], m[0, 0]]])