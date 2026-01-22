from collections import namedtuple, defaultdict
import operator
import warnings
from functools import partial
import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import (typing, utils, types, ir, debuginfo, funcdesc,
from numba.core.errors import (LoweringError, new_error_context, TypingError,
from numba.core.funcdesc import default_mangler
from numba.core.environment import Environment
from numba.core.analysis import compute_use_defs, must_use_alloca
from numba.misc.firstlinefinder import get_func_body_first_lineno
def _lower_call_normal(self, fnty, expr, signature):
    self.debug_print('# calling normal function: {0}'.format(fnty))
    self.debug_print('# signature: {0}'.format(signature))
    if isinstance(fnty, types.ObjModeDispatcher):
        argvals = expr.func.args
    else:
        argvals = self.fold_call_args(fnty, signature, expr.args, expr.vararg, expr.kws)
    tname = expr.target
    if tname is not None:
        from numba.core.target_extension import resolve_dispatcher_from_str
        disp = resolve_dispatcher_from_str(tname)
        hw_ctx = disp.targetdescr.target_context
        impl = hw_ctx.get_function(fnty, signature)
    else:
        impl = self.context.get_function(fnty, signature)
    if signature.recvr:
        the_self = self.loadvar(expr.func.name)
        argvals = [the_self] + list(argvals)
    res = impl(self.builder, argvals, self.loc)
    return res