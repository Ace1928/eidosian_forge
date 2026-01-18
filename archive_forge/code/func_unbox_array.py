from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@unbox(types.Array)
def unbox_array(typ, obj, c):
    """
    Convert a Numpy array object to a native array structure.
    """
    nativearycls = c.context.make_array(typ)
    nativeary = nativearycls(c.context, c.builder)
    aryptr = nativeary._getpointer()
    ptr = c.builder.bitcast(aryptr, c.pyapi.voidptr)
    if c.context.enable_nrt:
        errcode = c.pyapi.nrt_adapt_ndarray_from_python(obj, ptr)
    else:
        errcode = c.pyapi.numba_array_adaptor(obj, ptr)
    try:
        expected_itemsize = numpy_support.as_dtype(typ.dtype).itemsize
    except NumbaNotImplementedError:
        itemsize_mismatch = cgutils.false_bit
    else:
        expected_itemsize = nativeary.itemsize.type(expected_itemsize)
        itemsize_mismatch = c.builder.icmp_unsigned('!=', nativeary.itemsize, expected_itemsize)
    failed = c.builder.or_(cgutils.is_not_null(c.builder, errcode), itemsize_mismatch)
    with c.builder.if_then(failed, likely=False):
        c.pyapi.err_set_string('PyExc_TypeError', "can't unbox array from PyObject into native value.  The object maybe of a different type")
    return NativeValue(c.builder.load(aryptr), is_error=failed)