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
def user_prop_fresh(ctx, _new_ctx):
    _prop_closures.set_threaded()
    prop = _prop_closures.get(ctx)
    nctx = Context()
    Z3_del_context(nctx.ctx)
    new_ctx = to_ContextObj(_new_ctx)
    nctx.ctx = new_ctx
    nctx.eh = Z3_set_error_handler(new_ctx, z3_error_handler)
    nctx.owner = False
    new_prop = prop.fresh(nctx)
    _prop_closures.set(new_prop.id, new_prop)
    return new_prop.id