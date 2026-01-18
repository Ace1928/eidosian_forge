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
def timedelta_sign_impl(context, builder, sig, args):
    """
    np.sign(timedelta64)
    """
    val, = args
    ret = alloc_timedelta_result(builder)
    zero = Constant(TIMEDELTA64, 0)
    with builder.if_else(builder.icmp_signed('>', val, zero)) as (gt_zero, le_zero):
        with gt_zero:
            builder.store(Constant(TIMEDELTA64, 1), ret)
        with le_zero:
            with builder.if_else(builder.icmp_unsigned('==', val, zero)) as (eq_zero, lt_zero):
                with eq_zero:
                    builder.store(Constant(TIMEDELTA64, 0), ret)
                with lt_zero:
                    builder.store(Constant(TIMEDELTA64, -1), ret)
    res = builder.load(ret)
    return impl_ret_untracked(context, builder, sig.return_type, res)