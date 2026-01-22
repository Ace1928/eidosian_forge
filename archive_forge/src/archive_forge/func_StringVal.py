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
def StringVal(s, ctx=None):
    """create a string expression"""
    s = ''.join((str(ch) if 32 <= ord(ch) and ord(ch) < 127 else '\\u{%x}' % ord(ch) for ch in s))
    ctx = _get_ctx(ctx)
    return SeqRef(Z3_mk_string(ctx.ref(), s), ctx)