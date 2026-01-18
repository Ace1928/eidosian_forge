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
def try_static_impl(tys, args):
    if any((a is ir.UNDEFINED for a in args)):
        return None
    try:
        if isinstance(op, types.Function):
            static_sig = op.get_call_type(self.context.typing_context, tys, {})
        else:
            static_sig = typing.signature(signature.return_type, *tys)
    except TypingError:
        return None
    try:
        static_impl = self.context.get_function(op, static_sig)
        return static_impl(self.builder, args)
    except NotImplementedError:
        return None