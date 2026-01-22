from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
@staticmethod
def _three_perm_sign(v0, v1, v2):
    """
        Returns the sign of the permutation necessary to bring the three
        elements v0, v1, v2 in order.
        """
    if v0 < v2 and v2 < v1:
        return -1
    if v1 < v0 and v0 < v2:
        return -1
    if v2 < v1 and v1 < v0:
        return -1
    return +1