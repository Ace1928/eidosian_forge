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
def _coerce_char(ch, ctx=None):
    if isinstance(ch, str):
        ctx = _get_ctx(ctx)
        ch = CharVal(ch, ctx)
    if not is_expr(ch):
        raise Z3Exception('Character expression expected')
    return ch