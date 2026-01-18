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
def vararg_to_tuple(context, builder, sig, args):
    aryty = sig.args[0]
    dimtys = sig.args[1:]
    ary = args[0]
    dims = args[1:]
    dims = [context.cast(builder, val, ty, types.intp) for ty, val in zip(dimtys, dims)]
    shape = cgutils.pack_array(builder, dims, dims[0].type)
    shapety = types.UniTuple(dtype=types.intp, count=len(dims))
    new_sig = typing.signature(sig.return_type, aryty, shapety)
    new_args = (ary, shape)
    return (new_sig, new_args)