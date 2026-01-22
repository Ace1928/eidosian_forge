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
def _gauss_impl(state, loc_preprocessor, scale_preprocessor):

    def _impl(context, builder, sig, args):
        ty = sig.return_type
        llty = context.get_data_type(ty)
        _random = {'py': random.random, 'np': np.random.random}[state]
        state_ptr = get_state_ptr(context, builder, state)
        ret = cgutils.alloca_once(builder, llty, name='result')
        gauss_ptr = get_gauss_ptr(builder, state_ptr)
        has_gauss_ptr = get_has_gauss_ptr(builder, state_ptr)
        has_gauss = cgutils.is_true(builder, builder.load(has_gauss_ptr))
        with builder.if_else(has_gauss) as (then, otherwise):
            with then:
                builder.store(builder.load(gauss_ptr), ret)
                builder.store(const_int(0), has_gauss_ptr)
            with otherwise:
                pair = context.compile_internal(builder, _gauss_pair_impl(_random), signature(types.UniTuple(ty, 2)), ())
                first, second = cgutils.unpack_tuple(builder, pair, 2)
                builder.store(first, gauss_ptr)
                builder.store(second, ret)
                builder.store(const_int(1), has_gauss_ptr)
        mu, sigma = args
        return builder.fadd(loc_preprocessor(builder, mu), builder.fmul(scale_preprocessor(builder, sigma), builder.load(ret)))
    return _impl