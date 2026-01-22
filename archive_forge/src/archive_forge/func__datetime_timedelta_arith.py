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
def _datetime_timedelta_arith(ll_op_name):

    def impl(context, builder, dt_arg, dt_unit, td_arg, td_unit, ret_unit):
        ret = alloc_timedelta_result(builder)
        with cgutils.if_likely(builder, are_not_nat(builder, [dt_arg, td_arg])):
            dt_arg = convert_datetime_for_arith(builder, dt_arg, dt_unit, ret_unit)
            td_factor = npdatetime_helpers.get_timedelta_conversion_factor(td_unit, ret_unit)
            td_arg = scale_by_constant(builder, td_arg, td_factor)
            ret_val = getattr(builder, ll_op_name)(dt_arg, td_arg)
            builder.store(ret_val, ret)
        return builder.load(ret)
    return impl