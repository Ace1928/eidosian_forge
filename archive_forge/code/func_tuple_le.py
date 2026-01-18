import operator
from numba.core.imputils import (lower_builtin, lower_getattr_generic,
from numba.core import typing, types, cgutils
from numba.core.extending import overload_method, overload, intrinsic
@lower_builtin(operator.le, types.BaseTuple, types.BaseTuple)
def tuple_le(context, builder, sig, args):
    res = tuple_cmp_ordered(context, builder, operator.le, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)