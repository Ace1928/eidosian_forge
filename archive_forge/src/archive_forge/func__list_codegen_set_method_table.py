import operator
from enum import IntEnum
from llvmlite import ir
from numba.core.extending import (
from numba.core.imputils import iternext_impl
from numba.core import types, cgutils
from numba.core.types import (
from numba.core.imputils import impl_ret_borrowed, RefType
from numba.core.errors import TypingError
from numba.core import typing
from numba.typed.typedobjectutils import (_as_bytes, _cast, _nonoptional,
from numba.cpython import listobj
def _list_codegen_set_method_table(context, builder, lp, itemty):
    vtablety = ir.LiteralStructType([ll_voidptr_type, ll_voidptr_type])
    setmethod_fnty = ir.FunctionType(ir.VoidType(), [ll_list_type, vtablety.as_pointer()])
    setmethod_fn = cgutils.get_or_insert_function(builder.module, setmethod_fnty, 'numba_list_set_method_table')
    vtable = cgutils.alloca_once(builder, vtablety, zfill=True)
    item_incref_ptr = cgutils.gep_inbounds(builder, vtable, 0, 0)
    item_decref_ptr = cgutils.gep_inbounds(builder, vtable, 0, 1)
    dm_item = context.data_model_manager[itemty]
    if dm_item.contains_nrt_meminfo():
        item_incref, item_decref = _get_incref_decref(context, builder.module, dm_item, 'list')
        builder.store(builder.bitcast(item_incref, item_incref_ptr.type.pointee), item_incref_ptr)
        builder.store(builder.bitcast(item_decref, item_decref_ptr.type.pointee), item_decref_ptr)
    builder.call(setmethod_fn, [lp, vtable])