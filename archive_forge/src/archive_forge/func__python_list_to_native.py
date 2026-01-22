from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
def _python_list_to_native(typ, obj, c, size, listptr, errorptr):
    """
    Construct a new native list from a Python list.
    """

    def check_element_type(nth, itemobj, expected_typobj):
        typobj = nth.typeof(itemobj)
        with c.builder.if_then(cgutils.is_null(c.builder, typobj), likely=False):
            c.builder.store(cgutils.true_bit, errorptr)
            loop.do_break()
        type_mismatch = c.builder.icmp_signed('!=', typobj, expected_typobj)
        with c.builder.if_then(type_mismatch, likely=False):
            c.builder.store(cgutils.true_bit, errorptr)
            c.pyapi.err_format('PyExc_TypeError', "can't unbox heterogeneous list: %S != %S", expected_typobj, typobj)
            c.pyapi.decref(typobj)
            loop.do_break()
        c.pyapi.decref(typobj)
    ok, list = listobj.ListInstance.allocate_ex(c.context, c.builder, typ, size)
    with c.builder.if_else(ok, likely=True) as (if_ok, if_not_ok):
        with if_ok:
            list.size = size
            zero = ir.Constant(size.type, 0)
            with c.builder.if_then(c.builder.icmp_signed('>', size, zero), likely=True):
                with _NumbaTypeHelper(c) as nth:
                    expected_typobj = nth.typeof(c.pyapi.list_getitem(obj, zero))
                    with cgutils.for_range(c.builder, size) as loop:
                        itemobj = c.pyapi.list_getitem(obj, loop.index)
                        check_element_type(nth, itemobj, expected_typobj)
                        native = c.unbox(typ.dtype, itemobj)
                        with c.builder.if_then(native.is_error, likely=False):
                            c.builder.store(cgutils.true_bit, errorptr)
                            loop.do_break()
                        list.setitem(loop.index, native.value, incref=False)
                    c.pyapi.decref(expected_typobj)
            if typ.reflected:
                list.parent = obj
            with c.builder.if_then(c.builder.not_(c.builder.load(errorptr)), likely=False):
                c.pyapi.object_set_private_data(obj, list.meminfo)
            list.set_dirty(False)
            c.builder.store(list.value, listptr)
        with if_not_ok:
            c.builder.store(cgutils.true_bit, errorptr)
    with c.builder.if_then(c.builder.load(errorptr)):
        c.context.nrt.decref(c.builder, typ, list.value)