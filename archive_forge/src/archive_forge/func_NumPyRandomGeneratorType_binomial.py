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
@overload_method(types.NumPyRandomGeneratorType, 'binomial')
def NumPyRandomGeneratorType_binomial(inst, n, p, size=None):
    check_types(n, [types.Float, types.Integer, int, float], 'n')
    check_types(p, [types.Float, types.Integer, int, float], 'p')
    if isinstance(size, types.Omitted):
        size = size.value
    if is_nonelike(size):

        def impl(inst, n, p, size=None):
            return np.int64(random_binomial(inst.bit_generator, n, p))
        return impl
    else:
        check_size(size)

        def impl(inst, n, p, size=None):
            out = np.empty(size, dtype=np.int64)
            for i in np.ndindex(size):
                out[i] = random_binomial(inst.bit_generator, n, p)
            return out
        return impl