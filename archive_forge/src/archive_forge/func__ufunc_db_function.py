import math
import sys
import itertools
from collections import namedtuple
import llvmlite.ir as ir
import numpy as np
import operator
from numba.np import arrayobj, ufunc_db, numpy_support
from numba.core.imputils import Registry, impl_ret_new_ref, force_error_model
from numba.core import typing, types, utils, cgutils, callconv
from numba.np.numpy_support import (
from numba.core.typing import npydecl
from numba.core.extending import overload, intrinsic
from numba.core import errors
from numba.cpython import builtins
def _ufunc_db_function(ufunc):
    """Use the ufunc loop type information to select the code generation
    function from the table provided by the dict_of_kernels. The dict
    of kernels maps the loop identifier to a function with the
    following signature: (context, builder, signature, args).

    The loop type information has the form 'AB->C'. The letters to the
    left of '->' are the input types (specified as NumPy letter
    types).  The letters to the right of '->' are the output
    types. There must be 'ufunc.nin' letters to the left of '->', and
    'ufunc.nout' letters to the right.

    For example, a binary float loop resulting in a float, will have
    the following signature: 'ff->f'.

    A given ufunc implements many loops. The list of loops implemented
    for a given ufunc can be accessed using the 'types' attribute in
    the ufunc object. The NumPy machinery selects the first loop that
    fits a given calling signature (in our case, what we call the
    outer_sig). This logic is mimicked by 'ufunc_find_matching_loop'.
    """

    class _KernelImpl(_Kernel):

        def __init__(self, context, builder, outer_sig):
            super(_KernelImpl, self).__init__(context, builder, outer_sig)
            loop = ufunc_find_matching_loop(ufunc, outer_sig.args + tuple(_unpack_output_types(ufunc, outer_sig)))
            self.fn = context.get_ufunc_info(ufunc).get(loop.ufunc_sig)
            self.inner_sig = _ufunc_loop_sig(loop.outputs, loop.inputs)
            if self.fn is None:
                msg = "Don't know how to lower ufunc '{0}' for loop '{1}'"
                raise NotImplementedError(msg.format(ufunc.__name__, loop))

        def generate(self, *args):
            isig = self.inner_sig
            osig = self.outer_sig
            cast_args = [self.cast(val, inty, outty) for val, inty, outty in zip(args, osig.args, isig.args)]
            with force_error_model(self.context, 'numpy'):
                res = self.fn(self.context, self.builder, isig, cast_args)
            dmm = self.context.data_model_manager
            res = dmm[isig.return_type].from_return(self.builder, res)
            return self.cast(res, isig.return_type, osig.return_type)
    return _KernelImpl