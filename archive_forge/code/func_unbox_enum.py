from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@unbox(types.EnumMember)
def unbox_enum(typ, obj, c):
    """
    Convert an enum member's value to its native value.
    """
    valobj = c.pyapi.object_getattr_string(obj, 'value')
    return c.unbox(typ.dtype, valobj)