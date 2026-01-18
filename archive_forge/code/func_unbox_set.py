from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@unbox(types.Set)
def unbox_set(typ, obj, c):
    """
    Convert set *obj* to a native set.

    If set was previously unboxed, we reuse the existing native set
    to ensure consistency.
    """
    size = c.pyapi.set_size(obj)
    errorptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    setptr = cgutils.alloca_once(c.builder, c.context.get_value_type(typ))
    ptr = c.pyapi.object_get_private_data(obj)
    with c.builder.if_else(cgutils.is_not_null(c.builder, ptr)) as (has_meminfo, otherwise):
        with has_meminfo:
            inst = setobj.SetInstance.from_meminfo(c.context, c.builder, typ, ptr)
            if typ.reflected:
                inst.parent = obj
            c.builder.store(inst.value, setptr)
        with otherwise:
            _python_set_to_native(typ, obj, c, size, setptr, errorptr)

    def cleanup():
        c.pyapi.object_reset_private_data(obj)
    return NativeValue(c.builder.load(setptr), is_error=c.builder.load(errorptr), cleanup=cleanup)