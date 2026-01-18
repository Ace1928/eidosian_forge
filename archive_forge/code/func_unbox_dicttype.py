from collections.abc import MutableMapping, Iterable, Mapping
from numba.core.types import DictType
from numba.core.imputils import numba_typeref_ctor
from numba import njit, typeof
from numba.core import types, errors, config, cgutils
from numba.core.extending import (
from numba.typed import dictobject
from numba.core.typing import signature
@unbox(types.DictType)
def unbox_dicttype(typ, val, c):
    context = c.context
    dict_type = c.pyapi.unserialize(c.pyapi.serialize_object(Dict))
    valtype = c.pyapi.object_type(val)
    same_type = c.builder.icmp_unsigned('==', valtype, dict_type)
    with c.builder.if_else(same_type) as (then, orelse):
        with then:
            miptr = c.pyapi.object_getattr_string(val, '_opaque')
            mip_type = types.MemInfoPointer(types.voidptr)
            native = c.unbox(mip_type, miptr)
            mi = native.value
            argtypes = (mip_type, typeof(typ))

            def convert(mi, typ):
                return dictobject._from_meminfo(mi, typ)
            sig = signature(typ, *argtypes)
            nil_typeref = context.get_constant_null(argtypes[1])
            args = (mi, nil_typeref)
            is_error, dctobj = c.pyapi.call_jit_code(convert, sig, args)
            c.context.nrt.decref(c.builder, typ, dctobj)
            c.pyapi.decref(miptr)
            bb_unboxed = c.builder.basic_block
        with orelse:
            c.pyapi.err_format('PyExc_TypeError', "can't unbox a %S as a %S", valtype, dict_type)
            bb_else = c.builder.basic_block
    dctobj_res = c.builder.phi(dctobj.type)
    is_error_res = c.builder.phi(is_error.type)
    dctobj_res.add_incoming(dctobj, bb_unboxed)
    dctobj_res.add_incoming(dctobj.type(None), bb_else)
    is_error_res.add_incoming(is_error, bb_unboxed)
    is_error_res.add_incoming(cgutils.true_bit, bb_else)
    c.pyapi.decref(dict_type)
    c.pyapi.decref(valtype)
    return NativeValue(dctobj_res, is_error=is_error_res)