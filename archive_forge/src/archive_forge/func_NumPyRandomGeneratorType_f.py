import numpy as np
from numba.core import types
from numba.core.extending import overload_method, register_jitable
from numba.np.numpy_support import as_dtype, from_dtype
from numba.np.random.generator_core import next_float, next_double
from numba.np.numpy_support import is_nonelike
from numba.core.errors import TypingError
from numba.core.types.containers import Tuple, UniTuple
from numba.np.random.distributions import \
from numba.np.random import random_methods
@overload_method(types.NumPyRandomGeneratorType, 'f')
def NumPyRandomGeneratorType_f(inst, dfnum, dfden, size=None):
    check_types(dfnum, [types.Float, types.Integer, int, float], 'dfnum')
    check_types(dfden, [types.Float, types.Integer, int, float], 'dfden')
    if isinstance(size, types.Omitted):
        size = size.value
    if is_nonelike(size):

        def impl(inst, dfnum, dfden, size=None):
            return random_f(inst.bit_generator, dfnum, dfden)
        return impl
    else:
        check_size(size)

        def impl(inst, dfnum, dfden, size=None):
            out = np.empty(size)
            out_f = out.flat
            for i in range(out.size):
                out_f[i] = random_f(inst.bit_generator, dfnum, dfden)
            return out
        return impl