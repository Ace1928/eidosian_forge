from .mcomplex_base import *
from .kernel_structures import *
from . import t3mlite as t3m
from .t3mlite import ZeroSubsimplices, simplex
from .t3mlite import Corner, Perm4
from .t3mlite import V0, V1, V2, V3
from ..math_basics import prod
from functools import reduce
from ..sage_helper import _within_sage
def _matrix_L1_distance_to_kernel(m, snappeaM):
    return sum([abs(_diff_to_kernel(m[i, j], snappeaM[i, j])) for i in range(2) for j in range(2)])