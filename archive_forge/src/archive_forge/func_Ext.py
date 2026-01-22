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
def Ext(a, b):
    """Return extensionality index for one-dimensional arrays.
    >> a, b = Consts('a b', SetSort(IntSort()))
    >> Ext(a, b)
    Ext(a, b)
    """
    ctx = a.ctx
    if z3_debug():
        _z3_assert(is_array_sort(a) and (is_array(b) or b.is_lambda()), 'arguments must be arrays')
    return _to_expr_ref(Z3_mk_array_ext(ctx.ref(), a.as_ast(), b.as_ast()), ctx)