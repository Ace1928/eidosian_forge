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
def _lower_call_ExternalFunction(self, fnty, expr, signature):
    self.debug_print('# external function')
    argvals = self.fold_call_args(fnty, signature, expr.args, expr.vararg, expr.kws)
    fndesc = funcdesc.ExternalFunctionDescriptor(fnty.symbol, fnty.sig.return_type, fnty.sig.args)
    func = self.context.declare_external_function(self.builder.module, fndesc)
    return self.context.call_external_function(self.builder, func, fndesc.argtypes, argvals)