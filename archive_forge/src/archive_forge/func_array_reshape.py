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
@lower_builtin('array.reshape', types.Array, types.BaseTuple)
def array_reshape(context, builder, sig, args):
    aryty = sig.args[0]
    retty = sig.return_type
    shapety = sig.args[1]
    shape = args[1]
    ll_intp = context.get_value_type(types.intp)
    ll_shape = ir.ArrayType(ll_intp, shapety.count)
    ary = make_array(aryty)(context, builder, args[0])
    newshape = cgutils.alloca_once(builder, ll_shape)
    builder.store(shape, newshape)
    shape_ary_ty = types.Array(dtype=shapety.dtype, ndim=1, layout='C')
    shape_ary = make_array(shape_ary_ty)(context, builder)
    shape_itemsize = context.get_constant(types.intp, context.get_abi_sizeof(ll_intp))
    populate_array(shape_ary, data=builder.bitcast(newshape, ll_intp.as_pointer()), shape=[context.get_constant(types.intp, shapety.count)], strides=[shape_itemsize], itemsize=shape_itemsize, meminfo=None)
    size = ary.nitems
    context.compile_internal(builder, normalize_reshape_value, typing.signature(types.void, types.uintp, shape_ary_ty), [size, shape_ary._getvalue()])
    newnd = shapety.count
    newstrides = cgutils.alloca_once(builder, ll_shape)
    ok = _attempt_nocopy_reshape(context, builder, aryty, ary, newnd, newshape, newstrides)
    fail = builder.icmp_unsigned('==', ok, ok.type(0))
    with builder.if_then(fail):
        msg = 'incompatible shape for array'
        context.call_conv.return_user_exc(builder, NotImplementedError, (msg,))
    ret = make_array(retty)(context, builder)
    populate_array(ret, data=ary.data, shape=builder.load(newshape), strides=builder.load(newstrides), itemsize=ary.itemsize, meminfo=ary.meminfo, parent=ary.parent)
    res = ret._getvalue()
    return impl_ret_borrowed(context, builder, sig.return_type, res)