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
def _np_hstack(typingctx, tup):
    ret = NdStack_typer(typingctx, 'np.hstack', tup, 1)
    sig = ret(tup)

    def codegen(context, builder, sig, args):
        tupty = sig.args[0]
        ndim = tupty[0].ndim
        if ndim == 0:
            axis = context.get_constant(types.intp, 0)
            return _np_stack_common(context, builder, sig, args, axis)
        else:
            axis = 0 if ndim == 1 else 1

            def np_hstack_impl(arrays):
                return np.concatenate(arrays, axis=axis)
            return context.compile_internal(builder, np_hstack_impl, sig, args)
    return (sig, codegen)