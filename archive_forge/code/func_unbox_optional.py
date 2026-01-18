from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@unbox(types.Optional)
def unbox_optional(typ, obj, c):
    """
    Convert object *obj* to a native optional structure.
    """
    noneval = c.context.make_optional_none(c.builder, typ.type)
    is_not_none = c.builder.icmp_signed('!=', obj, c.pyapi.borrow_none())
    retptr = cgutils.alloca_once(c.builder, noneval.type)
    errptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    with c.builder.if_else(is_not_none) as (then, orelse):
        with then:
            native = c.unbox(typ.type, obj)
            just = c.context.make_optional_value(c.builder, typ.type, native.value)
            c.builder.store(just, retptr)
            c.builder.store(native.is_error, errptr)
        with orelse:
            c.builder.store(noneval, retptr)
    if native.cleanup is not None:

        def cleanup():
            with c.builder.if_then(is_not_none):
                native.cleanup()
    else:
        cleanup = None
    ret = c.builder.load(retptr)
    return NativeValue(ret, is_error=c.builder.load(errptr), cleanup=cleanup)