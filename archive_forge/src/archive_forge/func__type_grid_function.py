from llvmlite import ir
from numba import cuda, types
from numba.core import cgutils
from numba.core.errors import RequireLiteralValue
from numba.core.typing import signature
from numba.core.extending import overload_attribute
from numba.cuda import nvvmutils
from numba.cuda.extending import intrinsic
def _type_grid_function(ndim):
    val = ndim.literal_value
    if val == 1:
        restype = types.int64
    elif val in (2, 3):
        restype = types.UniTuple(types.int64, val)
    else:
        raise ValueError('argument can only be 1, 2, 3')
    return signature(restype, types.int32)