import math
import numbers
import numpy as np
import operator
from llvmlite import ir
from llvmlite.ir import Constant
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core import typing, types, utils, errors, cgutils, optional
from numba.core.extending import intrinsic, overload_method
from numba.cpython.unsafe.numbers import viewer
def _int_divmod_impl(context, builder, sig, args, zerodiv_message):
    va, vb = args
    ta, tb = sig.args
    ty = sig.return_type
    if isinstance(ty, types.UniTuple):
        ty = ty.dtype
    a = context.cast(builder, va, ta, ty)
    b = context.cast(builder, vb, tb, ty)
    quot = cgutils.alloca_once(builder, a.type, name='quot')
    rem = cgutils.alloca_once(builder, a.type, name='rem')
    with builder.if_else(cgutils.is_scalar_zero(builder, b), likely=False) as (if_zero, if_non_zero):
        with if_zero:
            if not context.error_model.fp_zero_division(builder, (zerodiv_message,)):
                builder.store(b, quot)
                builder.store(b, rem)
        with if_non_zero:
            q, r = int_divmod(context, builder, ty, a, b)
            builder.store(q, quot)
            builder.store(r, rem)
    return (quot, rem)