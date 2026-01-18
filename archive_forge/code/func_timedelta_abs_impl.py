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
@lower_builtin(abs, types.NPTimedelta)
def timedelta_abs_impl(context, builder, sig, args):
    val, = args
    ret = alloc_timedelta_result(builder)
    with builder.if_else(cgutils.is_scalar_neg(builder, val)) as (then, otherwise):
        with then:
            builder.store(builder.neg(val), ret)
        with otherwise:
            builder.store(val, ret)
    res = builder.load(ret)
    return impl_ret_untracked(context, builder, sig.return_type, res)