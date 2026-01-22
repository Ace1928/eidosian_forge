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
def SetHasSize(a, k):
    ctx = a.ctx
    k = _py2expr(k, ctx)
    return _to_expr_ref(Z3_mk_set_has_size(ctx.ref(), a.as_ast(), k.as_ast()), ctx)