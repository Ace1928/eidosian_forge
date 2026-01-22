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
def _lower_call_ObjModeDispatcher(self, fnty, expr, signature):
    from numba.core.pythonapi import ObjModeUtils
    self.init_pyapi()
    gil_state = self.pyapi.gil_ensure()
    argnames = [a.name for a in expr.args]
    argtypes = [self.typeof(a) for a in argnames]
    argvalues = [self.loadvar(a) for a in argnames]
    for v, ty in zip(argvalues, argtypes):
        self.incref(ty, v)
    argobjs = [self.pyapi.from_native_value(atyp, aval, self.env_manager) for atyp, aval in zip(argtypes, argvalues)]
    callee = ObjModeUtils(self.pyapi).load_dispatcher(fnty, argtypes)
    ret_obj = self.pyapi.call_function_objargs(callee, argobjs)
    has_exception = cgutils.is_null(self.builder, ret_obj)
    with self.builder.if_else(has_exception) as (then, orelse):
        with then:
            for obj in argobjs:
                self.pyapi.decref(obj)
            self.pyapi.gil_release(gil_state)
            self.call_conv.return_exc(self.builder)
        with orelse:
            native = self.pyapi.to_native_value(fnty.dispatcher.output_types, ret_obj)
            output = native.value
            self.pyapi.decref(ret_obj)
            for obj in argobjs:
                self.pyapi.decref(obj)
            if callable(native.cleanup):
                native.cleanup()
            self.pyapi.gil_release(gil_state)
            with self.builder.if_then(native.is_error):
                self.call_conv.return_exc(self.builder)
            return output