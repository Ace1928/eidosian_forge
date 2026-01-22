from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
@staticmethod
def _make_second(sides, x):
    """
        Cyclically rotate sides = (a,b,c) so that x is the second entry"
        """
    i = (sides.index(x) + 2) % len(sides)
    return sides[i:] + sides[:i]