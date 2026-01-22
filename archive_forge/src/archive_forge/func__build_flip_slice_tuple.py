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
@intrinsic
def _build_flip_slice_tuple(tyctx, sz):
    """ Creates a tuple of slices for np.flip indexing like
    `(slice(None, None, -1),) * sz` """
    if not isinstance(sz, types.IntegerLiteral):
        raise errors.RequireLiteralValue(sz)
    size = int(sz.literal_value)
    tuple_type = types.UniTuple(dtype=types.slice3_type, count=size)
    sig = tuple_type(sz)

    def codegen(context, builder, signature, args):

        def impl(length, empty_tuple):
            out = empty_tuple
            for i in range(length):
                out = tuple_setitem(out, i, slice(None, None, -1))
            return out
        inner_argtypes = [types.intp, tuple_type]
        inner_sig = typing.signature(tuple_type, *inner_argtypes)
        ll_idx_type = context.get_value_type(types.intp)
        empty_tuple = context.get_constant_undef(tuple_type)
        inner_args = [ll_idx_type(size), empty_tuple]
        res = context.compile_internal(builder, impl, inner_sig, inner_args)
        return res
    return (sig, codegen)