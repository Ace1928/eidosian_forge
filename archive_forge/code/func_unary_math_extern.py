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
def unary_math_extern(fn, f32extern, f64extern, int_restype=False):
    """
    Register implementations of Python function *fn* using the
    external function named *f32extern* and *f64extern* (for float32
    and float64 inputs, respectively).
    If *int_restype* is true, then the function's return value should be
    integral, otherwise floating-point.
    """
    f_restype = types.int64 if int_restype else None

    def float_impl(context, builder, sig, args):
        """
        Implement *fn* for a types.Float input.
        """
        [val] = args
        mod = builder.module
        input_type = sig.args[0]
        lty = context.get_value_type(input_type)
        func_name = {types.float32: f32extern, types.float64: f64extern}[input_type]
        fnty = llvmlite.ir.FunctionType(lty, [lty])
        fn = cgutils.insert_pure_function(builder.module, fnty, name=func_name)
        res = builder.call(fn, (val,))
        res = context.cast(builder, res, input_type, sig.return_type)
        return impl_ret_untracked(context, builder, sig.return_type, res)
    lower(fn, types.Float)(float_impl)
    unary_math_int_impl(fn, float_impl)
    return float_impl