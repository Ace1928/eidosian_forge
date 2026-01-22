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
def _mk_fp_tern(f, rm, a, b, c, ctx):
    ctx = _get_ctx(ctx)
    [a, b, c] = _coerce_fp_expr_list([a, b, c], ctx)
    if z3_debug():
        _z3_assert(is_fprm(rm), 'First argument must be a Z3 floating-point rounding mode expression')
        _z3_assert(is_fp(a) or is_fp(b) or is_fp(c), 'Second, third or fourth argument must be a Z3 floating-point expression')
    return FPRef(f(ctx.ref(), rm.as_ast(), a.as_ast(), b.as_ast(), c.as_ast()), ctx)