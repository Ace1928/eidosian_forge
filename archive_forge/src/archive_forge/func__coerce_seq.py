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
def _coerce_seq(s, ctx=None):
    if isinstance(s, str):
        ctx = _get_ctx(ctx)
        s = StringVal(s, ctx)
    if not is_expr(s):
        raise Z3Exception('Non-expression passed as a sequence')
    if not is_seq(s):
        raise Z3Exception('Non-sequence passed as a sequence')
    return s