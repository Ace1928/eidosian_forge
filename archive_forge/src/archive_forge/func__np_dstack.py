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
def _np_dstack(typingctx, tup):
    ret = NdStack_typer(typingctx, 'np.dstack', tup, 3)
    sig = ret(tup)

    def codegen(context, builder, sig, args):
        tupty = sig.args[0]
        retty = sig.return_type
        ndim = tupty[0].ndim
        if ndim == 0:

            def np_vstack_impl(arrays):
                return np.hstack(arrays).reshape(1, 1, -1)
            return context.compile_internal(builder, np_vstack_impl, sig, args)
        elif ndim == 1:
            axis = context.get_constant(types.intp, 1)
            stack_retty = retty.copy(ndim=retty.ndim - 1)
            stack_sig = typing.signature(stack_retty, *sig.args)
            stack_ret = _np_stack_common(context, builder, stack_sig, args, axis)
            axis = context.get_constant(types.intp, 0)
            expand_sig = typing.signature(retty, stack_retty)
            return expand_dims(context, builder, expand_sig, (stack_ret,), axis)
        elif ndim == 2:
            axis = context.get_constant(types.intp, 2)
            return _np_stack_common(context, builder, sig, args, axis)
        else:

            def np_vstack_impl(arrays):
                return np.concatenate(arrays, axis=2)
            return context.compile_internal(builder, np_vstack_impl, sig, args)
    return (sig, codegen)