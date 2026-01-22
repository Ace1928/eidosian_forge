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
@classmethod
def _size_of_list(cls, context, builder, list_ty, ll_list):
    tyctx = context.typing_context
    fnty = tyctx.resolve_value_type(len)
    sig = fnty.get_call_type(tyctx, (list_ty,), {})
    impl = context.get_function(fnty, sig)
    return impl(builder, (ll_list,))