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
def _coerce_expr_merge(s, a):
    if is_expr(a):
        s1 = a.sort()
        if s is None:
            return s1
        if s1.eq(s):
            return s
        elif s.subsort(s1):
            return s1
        elif s1.subsort(s):
            return s
        elif z3_debug():
            _z3_assert(s1.ctx == s.ctx, 'context mismatch')
            _z3_assert(False, 'sort mismatch')
    else:
        return s