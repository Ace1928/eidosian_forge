from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def _kronecker_delta(i, j):
    """
    Kronecker Delta, returns 1 if and only if i and j are equal, other 0.
    """
    if i == j:
        return 1
    else:
        return 0