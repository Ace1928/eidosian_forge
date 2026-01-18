from numba.extending import typeof_impl
from numba.extending import models, register_model
from numba.extending import unbox, NativeValue, box
from numba.core.imputils import lower_constant, lower_cast
from numba.core.ccallback import CFunc
from numba.core import cgutils
from llvmlite import ir
from numba.core import types
from numba.core.types import (FunctionType, UndefinedFunctionType,
from numba.core.dispatcher import Dispatcher
@unbox(FunctionType)
def unbox_function_type(typ, obj, c):
    typ = typ.get_precise()
    sfunc = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    addr = lower_get_wrapper_address(c.context, c.builder, obj, typ.signature, failure_mode='return_null')
    sfunc.addr = c.pyapi.long_as_voidptr(addr)
    c.pyapi.decref(addr)
    llty = c.context.get_value_type(types.voidptr)
    sfunc.pyaddr = c.builder.ptrtoint(obj, llty)
    return NativeValue(sfunc._getvalue())