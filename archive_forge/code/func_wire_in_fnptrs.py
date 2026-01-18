from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
def wire_in_fnptrs(name):
    interface_next_fn = c.pyapi.object_getattr_string(ctypes_binding, name)
    extra_refs.append(interface_next_fn)
    with cgutils.early_exit_if_null(c.builder, stack, interface_next_fn):
        handle_failure()
    args = c.pyapi.tuple_pack([interface_next_fn, ct_voidptr_ty])
    with cgutils.early_exit_if_null(c.builder, stack, args):
        handle_failure()
    interface_next_fn_casted = c.pyapi.call(ct_cast, args)
    interface_next_fn_casted_value = object_getattr_safely(interface_next_fn_casted, 'value')
    with cgutils.early_exit_if_null(c.builder, stack, interface_next_fn_casted_value):
        handle_failure()
    setattr(struct_ptr, f'fnptr_{name}', c.unbox(types.uintp, interface_next_fn_casted_value).value)