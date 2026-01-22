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
def _create_timedelta_comparison_impl(ll_op, default_value):

    def impl(context, builder, sig, args):
        [va, vb] = args
        [ta, tb] = sig.args
        ret = alloc_boolean_result(builder)
        with builder.if_else(are_not_nat(builder, [va, vb])) as (then, otherwise):
            with then:
                try:
                    norm_a, norm_b = normalize_timedeltas(context, builder, va, vb, ta, tb)
                except RuntimeError:
                    builder.store(default_value, ret)
                else:
                    builder.store(builder.icmp_unsigned(ll_op, norm_a, norm_b), ret)
            with otherwise:
                if ll_op == '!=':
                    builder.store(cgutils.true_bit, ret)
                else:
                    builder.store(cgutils.false_bit, ret)
        res = builder.load(ret)
        return impl_ret_untracked(context, builder, sig.return_type, res)
    return impl