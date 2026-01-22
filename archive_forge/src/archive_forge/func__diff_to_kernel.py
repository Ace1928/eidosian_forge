from .mcomplex_base import *
from .kernel_structures import *
from . import t3mlite as t3m
from .t3mlite import ZeroSubsimplices, simplex
from .t3mlite import Corner, Perm4
from .t3mlite import V0, V1, V2, V3
from ..math_basics import prod
from functools import reduce
from ..sage_helper import _within_sage
def _diff_to_kernel(value, snappeaValue):
    """
    The SnapPea kernel will always give us a number, but we might deal
    with a number or an interval.

    Cast to our numeric type so that we can compare.
    """
    CF = value.parent()
    return value - CF(snappeaValue)