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
def SeqSort(s):
    """Create a sequence sort over elements provided in the argument
    >>> s = SeqSort(IntSort())
    >>> s == Unit(IntVal(1)).sort()
    True
    """
    return SeqSortRef(Z3_mk_seq_sort(s.ctx_ref(), s.ast), s.ctx)