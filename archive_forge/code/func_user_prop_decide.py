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
def user_prop_decide(ctx, cb, t, idx, phase):
    prop = _prop_closures.get(ctx)
    old_cb = prop.cb
    prop.cb = cb
    t = _to_expr_ref(to_Ast(t_ref), prop.ctx())
    prop.decide(t, idx, phase)
    prop.cb = old_cb