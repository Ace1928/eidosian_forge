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
def Full(s):
    """Create the regular expression that accepts the universal language
    >>> e = Full(ReSort(SeqSort(IntSort())))
    >>> print(e)
    Full(ReSort(Seq(Int)))
    >>> e1 = Full(ReSort(StringSort()))
    >>> print(e1)
    Full(ReSort(String))
    """
    if isinstance(s, ReSortRef):
        return ReRef(Z3_mk_re_full(s.ctx_ref(), s.ast), s.ctx)
    raise Z3Exception('Non-sequence, non-regular expression sort passed to Full')