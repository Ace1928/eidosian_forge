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
def _lower_call_FunctionType(self, fnty, expr, signature):
    self.debug_print('# calling first-class function type')
    sig = types.unliteral(signature)
    if not fnty.check_signature(signature):
        raise UnsupportedError(f'mismatch of function types: expected {fnty} but got {types.FunctionType(sig)}')
    ftype = fnty.ftype
    argvals = self.fold_call_args(fnty, sig, expr.args, expr.vararg, expr.kws)
    func_ptr = self.__get_function_pointer(ftype, expr.func.name, sig=sig)
    res = self.builder.call(func_ptr, argvals, cconv=fnty.cconv)
    return res