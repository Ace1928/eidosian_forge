from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
def _native_set_to_python_list(typ, payload, c):
    """
    Create a Python list from a native set's items.
    """
    nitems = payload.used
    listobj = c.pyapi.list_new(nitems)
    ok = cgutils.is_not_null(c.builder, listobj)
    with c.builder.if_then(ok, likely=True):
        index = cgutils.alloca_once_value(c.builder, ir.Constant(nitems.type, 0))
        with payload._iterate() as loop:
            i = c.builder.load(index)
            item = loop.entry.key
            c.context.nrt.incref(c.builder, typ.dtype, item)
            itemobj = c.box(typ.dtype, item)
            c.pyapi.list_setitem(listobj, i, itemobj)
            i = c.builder.add(i, ir.Constant(i.type, 1))
            c.builder.store(i, index)
    return (ok, listobj)