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
def SetUnion(*args):
    """ Take the union of sets
    >>> a = Const('a', SetSort(IntSort()))
    >>> b = Const('b', SetSort(IntSort()))
    >>> SetUnion(a, b)
    union(a, b)
    """
    args = _get_args(args)
    ctx = _ctx_from_ast_arg_list(args)
    _args, sz = _to_ast_array(args)
    return ArrayRef(Z3_mk_set_union(ctx.ref(), sz, _args), ctx)