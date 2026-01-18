from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@unbox(types.SliceType)
def unbox_slice(typ, obj, c):
    """
    Convert object *obj* to a native slice structure.
    """
    from numba.cpython import slicing
    ok, start, stop, step = c.pyapi.slice_as_ints(obj)
    sli = c.context.make_helper(c.builder, typ)
    sli.start = start
    sli.stop = stop
    sli.step = step
    return NativeValue(sli._getvalue(), is_error=c.builder.not_(ok))