import operator
from numba.core.imputils import (lower_builtin, lower_getattr_generic,
from numba.core import typing, types, cgutils
from numba.core.extending import overload_method, overload, intrinsic
@lower_builtin(operator.gt, types.BaseTuple, types.BaseTuple)
def tuple_gt(context, builder, sig, args):
    res = tuple_cmp_ordered(context, builder, operator.gt, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)