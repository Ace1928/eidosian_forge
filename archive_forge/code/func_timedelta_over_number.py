import numpy as np
import operator
import llvmlite.ir
from llvmlite.ir import Constant
from numba.core import types, cgutils
from numba.core.cgutils import create_constant_array
from numba.core.imputils import (lower_builtin, lower_constant,
from numba.np import npdatetime_helpers, numpy_support, npyfuncs
from numba.extending import overload_method
from numba.core.config import IS_32BITS
from numba.core.errors import LoweringError
@lower_builtin(operator.truediv, types.NPTimedelta, types.Integer)
@lower_builtin(operator.itruediv, types.NPTimedelta, types.Integer)
@lower_builtin(operator.floordiv, types.NPTimedelta, types.Integer)
@lower_builtin(operator.ifloordiv, types.NPTimedelta, types.Integer)
@lower_builtin(operator.truediv, types.NPTimedelta, types.Float)
@lower_builtin(operator.itruediv, types.NPTimedelta, types.Float)
@lower_builtin(operator.floordiv, types.NPTimedelta, types.Float)
@lower_builtin(operator.ifloordiv, types.NPTimedelta, types.Float)
def timedelta_over_number(context, builder, sig, args):
    td_arg, number_arg = args
    number_type = sig.args[1]
    ret = alloc_timedelta_result(builder)
    ok = builder.and_(is_not_nat(builder, td_arg), builder.not_(cgutils.is_scalar_zero_or_nan(builder, number_arg)))
    with cgutils.if_likely(builder, ok):
        if isinstance(number_type, types.Float):
            val = builder.sitofp(td_arg, number_arg.type)
            val = builder.fdiv(val, number_arg)
            val = _cast_to_timedelta(context, builder, val)
        else:
            val = builder.sdiv(td_arg, number_arg)
        val = scale_timedelta(context, builder, val, sig.args[0], sig.return_type)
        builder.store(val, ret)
    res = builder.load(ret)
    return impl_ret_untracked(context, builder, sig.return_type, res)