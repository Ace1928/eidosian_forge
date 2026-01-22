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
def _do_concatenate(context, builder, axis, arrtys, arrs, arr_shapes, arr_strides, retty, ret_shapes):
    """
    Concatenate arrays along the given axis.
    """
    assert len(arrtys) == len(arrs) == len(arr_shapes) == len(arr_strides)
    zero = cgutils.intp_t(0)
    ret = _empty_nd_impl(context, builder, retty, ret_shapes)
    ret_strides = cgutils.unpack_tuple(builder, ret.strides)
    copy_offsets = []
    for arr_sh in arr_shapes:
        offset = zero
        for dim, (size, stride) in enumerate(zip(arr_sh, ret_strides)):
            is_axis = builder.icmp_signed('==', axis.type(dim), axis)
            addend = builder.mul(size, stride)
            offset = builder.select(is_axis, builder.add(offset, addend), offset)
        copy_offsets.append(offset)
    ret_data = ret.data
    for arrty, arr, arr_sh, arr_st, offset in zip(arrtys, arrs, arr_shapes, arr_strides, copy_offsets):
        arr_data = arr.data
        loop_nest = cgutils.loop_nest(builder, arr_sh, cgutils.intp_t, order=retty.layout)
        with loop_nest as indices:
            src_ptr = cgutils.get_item_pointer2(context, builder, arr_data, arr_sh, arr_st, arrty.layout, indices)
            val = load_item(context, builder, arrty, src_ptr)
            val = context.cast(builder, val, arrty.dtype, retty.dtype)
            dest_ptr = cgutils.get_item_pointer2(context, builder, ret_data, ret_shapes, ret_strides, retty.layout, indices)
            store_item(context, builder, retty, val, dest_ptr)
        ret_data = cgutils.pointer_add(builder, ret_data, offset)
    return ret