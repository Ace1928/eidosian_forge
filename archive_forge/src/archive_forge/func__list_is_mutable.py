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
@intrinsic
def _list_is_mutable(typingctx, l):
    """Wrap numba_list_is_mutable

    Returns the state of the is_mutable member
    """
    resty = types.int32
    sig = resty(l)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(ll_status, [ll_list_type])
        fn = cgutils.get_or_insert_function(builder.module, fnty, 'numba_list_is_mutable')
        [l] = args
        [tl] = sig.args
        lp = _container_get_data(context, builder, tl, l)
        n = builder.call(fn, [lp])
        return n
    return (sig, codegen)