import math
from numba import (config, cuda, float32, float64, uint32, int64, uint64,
import numpy as np
@jit(forceobj=_forceobj, looplift=_looplift, nopython=_nopython)
def xoroshiro128p_normal_float32(states, index):
    """Return a normally distributed float32 and advance ``states[index]``.

    The return value is drawn from a Gaussian of mean=0 and sigma=1 using the
    Box-Muller transform.  This advances the RNG sequence by two steps.

    :type states: 1D array, dtype=xoroshiro128p_dtype
    :param states: array of RNG states
    :type index: int64
    :param index: offset in states to update
    :rtype: float32
    """
    index = int64(index)
    u1 = xoroshiro128p_uniform_float32(states, index)
    u2 = xoroshiro128p_uniform_float32(states, index)
    z0 = math.sqrt(-float32(2.0) * math.log(u1)) * math.cos(TWO_PI_FLOAT32 * u2)
    return z0