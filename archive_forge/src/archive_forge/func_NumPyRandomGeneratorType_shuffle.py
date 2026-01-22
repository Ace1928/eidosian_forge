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
@overload_method(types.NumPyRandomGeneratorType, 'shuffle')
def NumPyRandomGeneratorType_shuffle(inst, x, axis=0):
    check_types(x, [types.Array], 'x')
    check_types(axis, [int, types.Integer], 'axis')

    def impl(inst, x, axis=0):
        if axis < 0:
            axis = axis + x.ndim
        if axis > x.ndim - 1 or axis < 0:
            raise IndexError('Axis is out of bounds for the given array')
        z = np.swapaxes(x, 0, axis)
        buf = np.empty_like(z[0, ...])
        for i in range(len(z) - 1, 0, -1):
            j = types.intp(random_methods.random_interval(inst.bit_generator, i))
            if i == j:
                continue
            buf[...] = z[j, ...]
            z[j, ...] = z[i, ...]
            z[i, ...] = buf
    return impl