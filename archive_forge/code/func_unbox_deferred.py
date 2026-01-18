from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@unbox(types.DeferredType)
def unbox_deferred(typ, obj, c):
    native_value = c.pyapi.to_native_value(typ.get(), obj)
    model = c.context.data_model_manager[typ]
    res = model.set(c.builder, model.make_uninitialized(), native_value.value)
    return NativeValue(res, is_error=native_value.is_error, cleanup=native_value.cleanup)