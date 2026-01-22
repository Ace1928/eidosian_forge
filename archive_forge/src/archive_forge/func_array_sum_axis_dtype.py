import math
from collections import namedtuple
import operator
import warnings
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.extending import overload, overload_method, register_jitable
from numba.np.numpy_support import (as_dtype, type_can_asarray, type_is_scalar,
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.np.arrayobj import (make_array, load_item, store_item,
from numba.np.linalg import ensure_blas
from numba.core.extending import intrinsic
from numba.core.errors import (RequireLiteralValue, TypingError,
from numba.cpython.unsafe.tuple import tuple_setitem
@lower_builtin(np.sum, types.Array, types.intp, types.DTypeSpec)
@lower_builtin(np.sum, types.Array, types.IntegerLiteral, types.DTypeSpec)
@lower_builtin('array.sum', types.Array, types.intp, types.DTypeSpec)
@lower_builtin('array.sum', types.Array, types.IntegerLiteral, types.DTypeSpec)
def array_sum_axis_dtype(context, builder, sig, args):
    retty = sig.return_type
    zero = getattr(retty, 'dtype', retty)(0)
    if getattr(retty, 'ndim', None) is None:
        op = np.take
    else:
        op = _array_sum_axis_nop
    [ty_array, ty_axis, ty_dtype] = sig.args
    is_axis_const = False
    const_axis_val = 0
    if isinstance(ty_axis, types.Literal):
        const_axis_val = ty_axis.literal_value
        if const_axis_val < 0:
            const_axis_val = ty_array.ndim + const_axis_val
        if const_axis_val < 0 or const_axis_val > ty_array.ndim:
            raise ValueError("'axis' entry is out of bounds")
        ty_axis = context.typing_context.resolve_value_type(const_axis_val)
        axis_val = context.get_constant(ty_axis, const_axis_val)
        args = (args[0], axis_val, args[2])
        sig = sig.replace(args=[ty_array, ty_axis, ty_dtype])
        is_axis_const = True
    gen_impl = gen_sum_axis_impl(is_axis_const, const_axis_val, op, zero)
    compiled = register_jitable(gen_impl)

    def array_sum_impl_axis(arr, axis, dtype):
        return compiled(arr, axis)
    res = context.compile_internal(builder, array_sum_impl_axis, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)