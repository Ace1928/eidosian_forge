from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@unbox(types.Complex)
def unbox_complex(typ, obj, c):
    c128 = c.context.make_complex(c.builder, types.complex128)
    ok = c.pyapi.complex_adaptor(obj, c128._getpointer())
    failed = cgutils.is_false(c.builder, ok)
    with cgutils.if_unlikely(c.builder, failed):
        c.pyapi.err_set_string('PyExc_TypeError', 'conversion to %s failed' % (typ,))
    if typ == types.complex64:
        cplx = c.context.make_complex(c.builder, typ)
        cplx.real = c.context.cast(c.builder, c128.real, types.float64, types.float32)
        cplx.imag = c.context.cast(c.builder, c128.imag, types.float64, types.float32)
    else:
        assert typ == types.complex128
        cplx = c128
    return NativeValue(cplx._getvalue(), is_error=failed)