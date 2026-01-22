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
@overload_method(types.NumPyRandomGeneratorType, 'integers')
def NumPyRandomGeneratorType_integers(inst, low, high, size=None, dtype=np.int64, endpoint=False):
    check_types(low, [types.Integer, types.Boolean, bool, int], 'low')
    check_types(high, [types.Integer, types.Boolean, bool, int], 'high')
    check_types(endpoint, [types.Boolean, bool], 'endpoint')
    if isinstance(size, types.Omitted):
        size = size.value
    if isinstance(dtype, types.Omitted):
        dtype = dtype.value
    if isinstance(dtype, type):
        nb_dt = from_dtype(np.dtype(dtype))
        _dtype = dtype
    elif isinstance(dtype, types.NumberClass):
        nb_dt = dtype
        _dtype = as_dtype(nb_dt)
    else:
        raise TypingError('Argument dtype is not one of the' + ' expected type(s): ' + 'np.int32, np.int64, np.int16, np.int8, np.uint32, np.uint64, np.uint16, np.uint8, np.bool_')
    if _dtype == np.bool_:
        int_func = random_methods.random_bounded_bool_fill
        lower_bound = -1
        upper_bound = 2
    else:
        try:
            i_info = np.iinfo(_dtype)
        except ValueError:
            raise TypingError('Argument dtype is not one of the' + ' expected type(s): ' + 'np.int32, np.int64, np.int16, np.int8, np.uint32, np.uint64, np.uint16, np.uint8, np.bool_')
        int_func = getattr(random_methods, f'random_bounded_uint{i_info.bits}_fill')
        lower_bound = i_info.min
        upper_bound = i_info.max
    if is_nonelike(size):

        def impl(inst, low, high, size=None, dtype=np.int64, endpoint=False):
            random_methods._randint_arg_check(low, high, endpoint, lower_bound, upper_bound)
            if not endpoint:
                high -= dtype(1)
                low = dtype(low)
                high = dtype(high)
                rng = high - low
                return int_func(inst.bit_generator, low, rng, 1, dtype)[0]
            else:
                low = dtype(low)
                high = dtype(high)
                rng = high - low
                return int_func(inst.bit_generator, low, rng, 1, dtype)[0]
        return impl
    else:
        check_size(size)

        def impl(inst, low, high, size=None, dtype=np.int64, endpoint=False):
            random_methods._randint_arg_check(low, high, endpoint, lower_bound, upper_bound)
            if not endpoint:
                high -= dtype(1)
                low = dtype(low)
                high = dtype(high)
                rng = high - low
                return int_func(inst.bit_generator, low, rng, size, dtype)
            else:
                low = dtype(low)
                high = dtype(high)
                rng = high - low
                return int_func(inst.bit_generator, low, rng, size, dtype)
        return impl