from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
class ReSortRef(SortRef):
    """Regular expression sort."""

    def basis(self):
        return _to_sort_ref(Z3_get_re_sort_basis(self.ctx_ref(), self.ast), self.ctx)