from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@unbox(types.UnicodeCharSeq)
def unbox_unicodecharseq(typ, obj, c):
    lty = c.context.get_value_type(typ)
    ok, buffer, size, kind, is_ascii, hashv = c.pyapi.string_as_string_size_and_kind(obj)
    with cgutils.if_likely(c.builder, ok):
        storage_size = ir.Constant(size.type, typ.count)
        size_fits = c.builder.icmp_unsigned('<=', size, storage_size)
        size = c.builder.select(size_fits, size, storage_size)
        null_string = ir.Constant(lty, None)
        outspace = cgutils.alloca_once_value(c.builder, null_string)
        cgutils.memcpy(c.builder, c.builder.bitcast(outspace, buffer.type), buffer, size)
    ret = c.builder.load(outspace)
    return NativeValue(ret, is_error=c.builder.not_(ok))