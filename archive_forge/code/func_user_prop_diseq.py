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
def user_prop_diseq(ctx, cb, x, y):
    prop = _prop_closures.get(ctx)
    old_cb = prop.cb
    prop.cb = cb
    x = _to_expr_ref(to_Ast(x), prop.ctx())
    y = _to_expr_ref(to_Ast(y), prop.ctx())
    prop.diseq(x, y)
    prop.cb = old_cb