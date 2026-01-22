from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def add_structures(self, one_cocycle=None):
    self._add_edge_dict()
    self._add_cusp_cross_sections(one_cocycle)