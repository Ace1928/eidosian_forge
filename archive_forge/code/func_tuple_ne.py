import operator
from numba.core.imputils import (lower_builtin, lower_getattr_generic,
from numba.core import typing, types, cgutils
from numba.core.extending import overload_method, overload, intrinsic
@lower_builtin(operator.ne, types.BaseTuple, types.BaseTuple)
def tuple_ne(context, builder, sig, args):
    res = builder.not_(tuple_eq(context, builder, sig, args))
    return impl_ret_untracked(context, builder, sig.return_type, res)