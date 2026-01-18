import math
from numba import (config, cuda, float32, float64, uint32, int64, uint64,
import numpy as np
@jit(forceobj=_forceobj, looplift=_looplift, nopython=_nopython)
def xoroshiro128p_uniform_float64(states, index):
    """Return a float64 in range [0.0, 1.0) and advance ``states[index]``.

    :type states: 1D array, dtype=xoroshiro128p_dtype
    :param states: array of RNG states
    :type index: int64
    :param index: offset in states to update
    :rtype: float64
    """
    index = int64(index)
    return uint64_to_unit_float64(xoroshiro128p_next(states, index))