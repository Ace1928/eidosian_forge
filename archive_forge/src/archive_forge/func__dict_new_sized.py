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
def _dict_new_sized(typingctx, n_keys, keyty, valty):
    """Wrap numba_dict_new_sized.

    Allocate a new dictionary object with enough space to hold
    *n_keys* keys without needing a resize.

    Parameters
    ----------
    keyty, valty: Type
        Type of the key and value, respectively.
    n_keys: int
        The number of keys to insert without needing a resize.
        A value of 0 creates a dict with minimum size.
    """
    resty = types.voidptr
    sig = resty(n_keys, keyty, valty)

    def codegen(context, builder, sig, args):
        n_keys = builder.bitcast(args[0], ll_ssize_t)
        ll_key = context.get_data_type(keyty.instance_type)
        ll_val = context.get_data_type(valty.instance_type)
        sz_key = context.get_abi_sizeof(ll_key)
        sz_val = context.get_abi_sizeof(ll_val)
        refdp = cgutils.alloca_once(builder, ll_dict_type, zfill=True)
        argtys = [ll_dict_type.as_pointer(), ll_ssize_t, ll_ssize_t, ll_ssize_t]
        fnty = ir.FunctionType(ll_status, argtys)
        fn = ir.Function(builder.module, fnty, 'numba_dict_new_sized')
        args = [refdp, n_keys, ll_ssize_t(sz_key), ll_ssize_t(sz_val)]
        status = builder.call(fn, args)
        allocated_failed_msg = 'Failed to allocate dictionary'
        _raise_if_error(context, builder, status, msg=allocated_failed_msg)
        dp = builder.load(refdp)
        return dp
    return (sig, codegen)