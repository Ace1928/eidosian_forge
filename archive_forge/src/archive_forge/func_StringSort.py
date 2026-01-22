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
def StringSort(ctx=None):
    """Create a string sort
    >>> s = StringSort()
    >>> print(s)
    String
    """
    ctx = _get_ctx(ctx)
    return SeqSortRef(Z3_mk_string_sort(ctx.ref()), ctx)