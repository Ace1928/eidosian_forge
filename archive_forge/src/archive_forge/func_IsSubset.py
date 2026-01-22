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
def IsSubset(a, b):
    """ Check if a is a subset of b
    >>> a = Const('a', SetSort(IntSort()))
    >>> b = Const('b', SetSort(IntSort()))
    >>> IsSubset(a, b)
    subset(a, b)
    """
    ctx = _ctx_from_ast_arg_list([a, b])
    return BoolRef(Z3_mk_set_subset(ctx.ref(), a.as_ast(), b.as_ast()), ctx)