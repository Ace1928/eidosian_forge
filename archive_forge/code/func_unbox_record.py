from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@unbox(types.Record)
def unbox_record(typ, obj, c):
    buf = c.pyapi.alloca_buffer()
    ptr = c.pyapi.extract_record_data(obj, buf)
    is_error = cgutils.is_null(c.builder, ptr)
    ltyp = c.context.get_value_type(typ)
    val = c.builder.bitcast(ptr, ltyp)

    def cleanup():
        c.pyapi.release_buffer(buf)
    return NativeValue(val, cleanup=cleanup, is_error=is_error)