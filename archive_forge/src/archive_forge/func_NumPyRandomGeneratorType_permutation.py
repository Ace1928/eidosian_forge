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
@overload_method(types.NumPyRandomGeneratorType, 'permutation')
def NumPyRandomGeneratorType_permutation(inst, x, axis=0):
    check_types(x, [types.Array, types.Integer], 'x')
    check_types(axis, [int, types.Integer], 'axis')
    IS_INT = isinstance(x, types.Integer)

    def impl(inst, x, axis=0):
        if IS_INT:
            new_arr = np.arange(x)
            inst.shuffle(new_arr)
        else:
            new_arr = x.copy()
            inst.shuffle(new_arr, axis=axis)
        return new_arr
    return impl