import functools
import math
import operator
from llvmlite import ir
from llvmlite.ir import Constant
import numpy as np
from numba import pndindex, literal_unroll
from numba.core import types, typing, errors, cgutils, extending
from numba.np.numpy_support import (as_dtype, from_dtype, carray, farray,
from numba.np.numpy_support import type_can_asarray, is_nonelike, numpy_version
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core.typing import signature
from numba.core.types import StringLiteral
from numba.core.extending import (register_jitable, overload, overload_method,
from numba.misc import quicksort, mergesort
from numba.cpython import slicing
from numba.cpython.unsafe.tuple import tuple_setitem, build_full_slice_tuple
from numba.core.extending import overload_classmethod
from numba.core.typing.npydecl import (parse_dtype as ty_parse_dtype,
def _np_concatenate(context, builder, arrtys, arrs, retty, axis):
    ndim = retty.ndim
    arrs = [make_array(aty)(context, builder, value=a) for aty, a in zip(arrtys, arrs)]
    axis = _normalize_axis(context, builder, 'np.concatenate', ndim, axis)
    arr_shapes = [cgutils.unpack_tuple(builder, arr.shape) for arr in arrs]
    arr_strides = [cgutils.unpack_tuple(builder, arr.strides) for arr in arrs]
    ret_shapes = [cgutils.alloca_once_value(builder, sh) for sh in arr_shapes[0]]
    for dim in range(ndim):
        is_axis = builder.icmp_signed('==', axis.type(dim), axis)
        ret_shape_ptr = ret_shapes[dim]
        ret_sh = builder.load(ret_shape_ptr)
        other_shapes = [sh[dim] for sh in arr_shapes[1:]]
        with builder.if_else(is_axis) as (on_axis, on_other_dim):
            with on_axis:
                sh = functools.reduce(builder.add, other_shapes + [ret_sh])
                builder.store(sh, ret_shape_ptr)
            with on_other_dim:
                is_ok = cgutils.true_bit
                for sh in other_shapes:
                    is_ok = builder.and_(is_ok, builder.icmp_signed('==', sh, ret_sh))
                with builder.if_then(builder.not_(is_ok), likely=False):
                    context.call_conv.return_user_exc(builder, ValueError, ('np.concatenate(): input sizes over dimension %d do not match' % dim,))
    ret_shapes = [builder.load(sh) for sh in ret_shapes]
    ret = _do_concatenate(context, builder, axis, arrtys, arrs, arr_shapes, arr_strides, retty, ret_shapes)
    return impl_ret_new_ref(context, builder, retty, ret._getvalue())