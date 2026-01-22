import math
import operator
import sys
import numpy as np
import llvmlite.ir
from llvmlite.ir import Constant
from numba.core.imputils import Registry, impl_ret_untracked
from numba import typeof
from numba.core import types, utils, config, cgutils
from numba.core.extending import overload
from numba.core.typing import signature
from numba.cpython.unsafe.numbers import trailing_zeros
def _unary_int_input_wrapper_impl(wrapped_impl):
    """
    Return an implementation factory to convert the single integral input
    argument to a float64, then defer to the *wrapped_impl*.
    """

    def implementer(context, builder, sig, args):
        val, = args
        input_type = sig.args[0]
        fpval = context.cast(builder, val, input_type, types.float64)
        inner_sig = signature(types.float64, types.float64)
        res = wrapped_impl(context, builder, inner_sig, (fpval,))
        return context.cast(builder, res, types.float64, sig.return_type)
    return implementer