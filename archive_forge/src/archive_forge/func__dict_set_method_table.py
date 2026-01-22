import ctypes
import operator
from enum import IntEnum
from llvmlite import ir
from numba import _helperlib
from numba.core.extending import (
from numba.core.imputils import iternext_impl, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.types import (
from numba.core.imputils import impl_ret_borrowed, RefType
from numba.core.errors import TypingError, LoweringError
from numba.core import typing
from numba.typed.typedobjectutils import (_as_bytes, _cast, _nonoptional,
@intrinsic
def _dict_set_method_table(typingctx, dp, keyty, valty):
    """Wrap numba_dict_set_method_table
    """
    resty = types.void
    sig = resty(dp, keyty, valty)

    def codegen(context, builder, sig, args):
        vtablety = ir.LiteralStructType([ll_voidptr_type, ll_voidptr_type, ll_voidptr_type, ll_voidptr_type, ll_voidptr_type])
        setmethod_fnty = ir.FunctionType(ir.VoidType(), [ll_dict_type, vtablety.as_pointer()])
        setmethod_fn = ir.Function(builder.module, setmethod_fnty, name='numba_dict_set_method_table')
        dp = args[0]
        vtable = cgutils.alloca_once(builder, vtablety, zfill=True)
        key_equal_ptr = cgutils.gep_inbounds(builder, vtable, 0, 0)
        key_incref_ptr = cgutils.gep_inbounds(builder, vtable, 0, 1)
        key_decref_ptr = cgutils.gep_inbounds(builder, vtable, 0, 2)
        val_incref_ptr = cgutils.gep_inbounds(builder, vtable, 0, 3)
        val_decref_ptr = cgutils.gep_inbounds(builder, vtable, 0, 4)
        dm_key = context.data_model_manager[keyty.instance_type]
        if dm_key.contains_nrt_meminfo():
            equal = _get_equal(context, builder.module, dm_key, 'dict_key')
            key_incref, key_decref = _get_incref_decref(context, builder.module, dm_key, 'dict_key')
            builder.store(builder.bitcast(equal, key_equal_ptr.type.pointee), key_equal_ptr)
            builder.store(builder.bitcast(key_incref, key_incref_ptr.type.pointee), key_incref_ptr)
            builder.store(builder.bitcast(key_decref, key_decref_ptr.type.pointee), key_decref_ptr)
        dm_val = context.data_model_manager[valty.instance_type]
        if dm_val.contains_nrt_meminfo():
            val_incref, val_decref = _get_incref_decref(context, builder.module, dm_val, 'dict_value')
            builder.store(builder.bitcast(val_incref, val_incref_ptr.type.pointee), val_incref_ptr)
            builder.store(builder.bitcast(val_decref, val_decref_ptr.type.pointee), val_decref_ptr)
        builder.call(setmethod_fn, [dp, vtable])
    return (sig, codegen)