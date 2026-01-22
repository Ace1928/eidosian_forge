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
@intrinsic
def _make_dtype_object(typingctx, desc):
    """Given a string or NumberClass description *desc*, returns the dtype object.
    """

    def from_nb_type(nb_type):
        return_type = types.DType(nb_type)
        sig = return_type(desc)

        def codegen(context, builder, signature, args):
            return context.get_dummy_value()
        return (sig, codegen)
    if isinstance(desc, types.Literal):
        nb_type = from_dtype(np.dtype(desc.literal_value))
        return from_nb_type(nb_type)
    elif isinstance(desc, types.functions.NumberClass):
        thestr = str(desc.dtype)
        nb_type = from_dtype(np.dtype(thestr))
        return from_nb_type(nb_type)