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
def _coerce_expr_list(alist, ctx=None):
    has_expr = False
    for a in alist:
        if is_expr(a):
            has_expr = True
            break
    if not has_expr:
        alist = [_py2expr(a, ctx) for a in alist]
    s = _reduce(_coerce_expr_merge, alist, None)
    return [s.cast(a) for a in alist]