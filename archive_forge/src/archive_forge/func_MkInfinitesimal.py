from .z3 import *
from .z3core import *
from .z3printer import *
def MkInfinitesimal(name='eps', ctx=None):
    ctx = z3.get_ctx(ctx)
    return RCFNum(Z3_rcf_mk_infinitesimal(ctx.ref()), ctx)