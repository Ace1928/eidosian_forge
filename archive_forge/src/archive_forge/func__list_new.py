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
def _list_new(typingctx, itemty, allocated):
    """Wrap numba_list_new.

    Allocate a new list object with zero capacity.

    Parameters
    ----------
    itemty: Type
        Type of the items
    allocated: int
        number of items to pre-allocate

    """
    resty = types.voidptr
    sig = resty(itemty, allocated)

    def codegen(context, builder, sig, args):
        error_handler = ErrorHandler(context)
        return _list_new_codegen(context, builder, itemty.instance_type, args[1], error_handler)
    return (sig, codegen)