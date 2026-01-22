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
def _create_datetime_comparison_impl(ll_op):

    def impl(context, builder, sig, args):
        va, vb = args
        ta, tb = sig.args
        unit_a = ta.unit
        unit_b = tb.unit
        ret_unit = npdatetime_helpers.get_best_unit(unit_a, unit_b)
        ret = alloc_boolean_result(builder)
        with builder.if_else(are_not_nat(builder, [va, vb])) as (then, otherwise):
            with then:
                norm_a = convert_datetime_for_arith(builder, va, unit_a, ret_unit)
                norm_b = convert_datetime_for_arith(builder, vb, unit_b, ret_unit)
                ret_val = builder.icmp_signed(ll_op, norm_a, norm_b)
                builder.store(ret_val, ret)
            with otherwise:
                if ll_op == '!=':
                    ret_val = cgutils.true_bit
                else:
                    ret_val = cgutils.false_bit
                builder.store(ret_val, ret)
        res = builder.load(ret)
        return impl_ret_untracked(context, builder, sig.return_type, res)
    return impl