from collections import namedtuple
import numpy as np
from llvmlite.ir import Constant, IRBuilder
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.caching import make_library_cache, NullCache
class _ScalarArgLoader(object):
    """
    Handle GFunc argument loading where a scalar type is used in the core
    function.
    Note: It still has a stride because the input to the gufunc can be an array
          for this argument.
    """

    def __init__(self, dtype, stride):
        self.dtype = dtype
        self.stride = stride

    def load(self, context, builder, data, ind):
        data = builder.gep(data, [builder.mul(ind, self.stride)])
        dptr = builder.bitcast(data, context.get_data_type(self.dtype).as_pointer())
        return builder.load(dptr)