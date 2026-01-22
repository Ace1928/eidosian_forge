from functools import wraps, partial
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.decorators import njit
from numba.core.pythonapi import box, unbox, NativeValue
from numba.core.typing.typeof import typeof_impl
from numba.experimental.jitclass import _box
def access_member(member_offset):
    offset = c.context.get_constant(types.uintp, member_offset)
    llvoidptr = ir.IntType(8).as_pointer()
    ptr = cgutils.pointer_add(c.builder, val, offset)
    casted = c.builder.bitcast(ptr, llvoidptr.as_pointer())
    return c.builder.load(casted)