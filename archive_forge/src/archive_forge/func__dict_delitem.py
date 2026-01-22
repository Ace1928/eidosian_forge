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
def _dict_delitem(typingctx, d, hk, ix):
    """Wrap numba_dict_delitem
    """
    resty = types.int32
    sig = resty(d, hk, types.intp)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(ll_status, [ll_dict_type, ll_hash, ll_ssize_t])
        [d, hk, ix] = args
        [td, thk, tix] = sig.args
        fn = cgutils.get_or_insert_function(builder.module, fnty, 'numba_dict_delitem')
        dp = _container_get_data(context, builder, td, d)
        status = builder.call(fn, [dp, hk, ix])
        return status
    return (sig, codegen)