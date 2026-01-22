from numba import typeof
from numba.core import types
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder
from numba.np.ufunc.sigparse import parse_signature
from numba.np.numpy_support import ufunc_find_matching_loop
from numba.core import serialize
import functools
def _get_ewise_dtypes(self, args):
    argtys = map(lambda x: typeof(x), args)
    tys = []
    for argty in argtys:
        if isinstance(argty, types.Array):
            tys.append(argty.dtype)
        else:
            tys.append(argty)
    return tys