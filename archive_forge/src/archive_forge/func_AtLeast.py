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
def AtLeast(*args):
    """Create an at-least Pseudo-Boolean k constraint.

    >>> a, b, c = Bools('a b c')
    >>> f = AtLeast(a, b, c, 2)
    """
    args = _get_args(args)
    if z3_debug():
        _z3_assert(len(args) > 1, 'Non empty list of arguments expected')
    ctx = _ctx_from_ast_arg_list(args)
    if z3_debug():
        _z3_assert(ctx is not None, 'At least one of the arguments must be a Z3 expression')
    args1 = _coerce_expr_list(args[:-1], ctx)
    k = args[-1]
    _args, sz = _to_ast_array(args1)
    return BoolRef(Z3_mk_atleast(ctx.ref(), sz, _args, k), ctx)