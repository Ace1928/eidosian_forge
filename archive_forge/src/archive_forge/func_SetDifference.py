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
def SetDifference(a, b):
    """ The set difference of a and b
    >>> a = Const('a', SetSort(IntSort()))
    >>> b = Const('b', SetSort(IntSort()))
    >>> SetDifference(a, b)
    setminus(a, b)
    """
    ctx = _ctx_from_ast_arg_list([a, b])
    return ArrayRef(Z3_mk_set_difference(ctx.ref(), a.as_ast(), b.as_ast()), ctx)