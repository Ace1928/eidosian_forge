from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@unbox(types.NumPyRandomGeneratorType)
def unbox_numpy_random_generator(typ, obj, c):
    """
    Here we're creating a NumPyRandomGeneratorType StructModel with following fields:
    * ('bit_generator', _bit_gen_type): The unboxed BitGenerator associated with
                                        this Generator object instance.
    * ('parent', types.pyobject): Pointer to the original Generator PyObject.
    * ('meminfo', types.MemInfoPointer(types.voidptr)): The information about the memory
        stored at the pointer (to the original Generator PyObject). This is useful for
        keeping track of reference counts within the Python runtime. Helps prevent cases
        where deletion happens in Python runtime without NRT being awareness of it. 
    """
    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    with ExitStack() as stack:
        struct_ptr = cgutils.create_struct_proxy(typ)(c.context, c.builder)
        bit_gen_inst = c.pyapi.object_getattr_string(obj, 'bit_generator')
        with cgutils.early_exit_if_null(c.builder, stack, bit_gen_inst):
            c.builder.store(cgutils.true_bit, is_error_ptr)
        unboxed = c.unbox(_bit_gen_type, bit_gen_inst).value
        struct_ptr.bit_generator = unboxed
        struct_ptr.parent = obj
        NULL = cgutils.voidptr_t(None)
        struct_ptr.meminfo = c.pyapi.nrt_meminfo_new_from_pyobject(NULL, obj)
        c.pyapi.decref(bit_gen_inst)
    return NativeValue(struct_ptr._getvalue(), is_error=c.builder.load(is_error_ptr))