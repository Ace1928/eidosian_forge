import math
import random
import numpy as np
from llvmlite import ir
from numba.core.cgutils import is_nonelike
from numba.core.extending import intrinsic, overload, register_jitable
from numba.core.imputils import (Registry, impl_ret_untracked,
from numba.core.typing import signature
from numba.core import types, cgutils
from numba.np import arrayobj
from numba.core.errors import NumbaTypeError
def _randrange_impl(context, builder, start, stop, step, ty, signed, state):
    state_ptr = get_state_ptr(context, builder, state)
    zero = ir.Constant(ty, 0)
    one = ir.Constant(ty, 1)
    nptr = cgutils.alloca_once(builder, ty, name='n')
    builder.store(builder.sub(stop, start), nptr)
    with builder.if_then(builder.icmp_signed('<', step, zero)):
        w = builder.add(builder.add(builder.load(nptr), step), one)
        n = builder.sdiv(w, step)
        builder.store(n, nptr)
    with builder.if_then(builder.icmp_signed('>', step, one)):
        w = builder.sub(builder.add(builder.load(nptr), step), one)
        n = builder.sdiv(w, step)
        builder.store(n, nptr)
    n = builder.load(nptr)
    with cgutils.if_unlikely(builder, builder.icmp_signed('<=', n, zero)):
        msg = 'empty range for randrange()'
        context.call_conv.return_user_exc(builder, ValueError, (msg,))
    fnty = ir.FunctionType(ty, [ty, cgutils.true_bit.type])
    fn = cgutils.get_or_insert_function(builder.function.module, fnty, 'llvm.ctlz.%s' % ty)
    nm1 = builder.sub(n, one) if state == 'np' else n
    nbits = builder.trunc(builder.call(fn, [nm1, cgutils.true_bit]), int32_t)
    nbits = builder.sub(ir.Constant(int32_t, ty.width), nbits)
    rptr = cgutils.alloca_once(builder, ty, name='r')

    def get_num():
        bbwhile = builder.append_basic_block('while')
        bbend = builder.append_basic_block('while.end')
        builder.branch(bbwhile)
        builder.position_at_end(bbwhile)
        r = get_next_int(context, builder, state_ptr, nbits, state == 'np')
        r = builder.trunc(r, ty)
        too_large = builder.icmp_signed('>=', r, n)
        builder.cbranch(too_large, bbwhile, bbend)
        builder.position_at_end(bbend)
        builder.store(r, rptr)
    if state == 'np':
        with builder.if_else(builder.icmp_signed('==', n, one)) as (is_one, is_not_one):
            with is_one:
                builder.store(zero, rptr)
            with is_not_one:
                get_num()
    else:
        get_num()
    return builder.add(start, builder.mul(builder.load(rptr), step))