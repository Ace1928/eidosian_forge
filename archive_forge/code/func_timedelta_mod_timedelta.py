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
def timedelta_mod_timedelta(context, builder, sig, args):
    [va, vb] = args
    [ta, tb] = sig.args
    not_nan = are_not_nat(builder, [va, vb])
    ll_ret_type = context.get_value_type(sig.return_type)
    ret = alloc_timedelta_result(builder)
    builder.store(NAT, ret)
    zero = Constant(ll_ret_type, 0)
    with cgutils.if_likely(builder, not_nan):
        va, vb = normalize_timedeltas(context, builder, va, vb, ta, tb)
        denom_ok = builder.not_(builder.icmp_signed('==', vb, zero))
        with cgutils.if_likely(builder, denom_ok):
            vapos = builder.icmp_signed('>', va, zero)
            vbpos = builder.icmp_signed('>', vb, zero)
            rem = builder.srem(va, vb)
            cond = builder.or_(builder.and_(vapos, vbpos), builder.icmp_signed('==', rem, zero))
            with builder.if_else(cond) as (then, otherwise):
                with then:
                    builder.store(rem, ret)
                with otherwise:
                    builder.store(builder.add(rem, vb), ret)
    res = builder.load(ret)
    return impl_ret_untracked(context, builder, sig.return_type, res)