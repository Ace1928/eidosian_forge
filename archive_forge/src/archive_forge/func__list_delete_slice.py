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
def _list_delete_slice(typingctx, l, start, stop, step):
    """Wrap numba_list_delete_slice
    """
    resty = types.int32
    sig = resty(l, start, stop, step)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(ll_status, [ll_list_type, ll_ssize_t, ll_ssize_t, ll_ssize_t])
        [l, start, stop, step] = args
        [tl, tstart, tstop, tstep] = sig.args
        fn = cgutils.get_or_insert_function(builder.module, fnty, 'numba_list_delete_slice')
        lp = _container_get_data(context, builder, tl, l)
        status = builder.call(fn, [lp, start, stop, step])
        return status
    return (sig, codegen)