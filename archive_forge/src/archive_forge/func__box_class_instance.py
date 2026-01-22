from functools import wraps, partial
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.decorators import njit
from numba.core.pythonapi import box, unbox, NativeValue
from numba.core.typing.typeof import typeof_impl
from numba.experimental.jitclass import _box
@box(types.ClassInstanceType)
def _box_class_instance(typ, val, c):
    meminfo, dataptr = cgutils.unpack_tuple(c.builder, val)
    box_subclassed = _specialize_box(typ)
    voidptr_boxcls = c.context.add_dynamic_addr(c.builder, id(box_subclassed), info='box_class_instance')
    box_cls = c.builder.bitcast(voidptr_boxcls, c.pyapi.pyobj)
    box = c.pyapi.call_function_objargs(box_cls, ())
    llvoidptr = ir.IntType(8).as_pointer()
    addr_meminfo = c.builder.bitcast(meminfo, llvoidptr)
    addr_data = c.builder.bitcast(dataptr, llvoidptr)

    def set_member(member_offset, value):
        offset = c.context.get_constant(types.uintp, member_offset)
        ptr = cgutils.pointer_add(c.builder, box, offset)
        casted = c.builder.bitcast(ptr, llvoidptr.as_pointer())
        c.builder.store(value, casted)
    set_member(_box.box_meminfoptr_offset, addr_meminfo)
    set_member(_box.box_dataptr_offset, addr_data)
    return box