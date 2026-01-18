import math
from numba import (config, cuda, float32, float64, uint32, int64, uint64,
import numpy as np
@jit(forceobj=_forceobj, looplift=_looplift, nopython=_nopython)
def xoroshiro128p_jump(states, index):
    """Advance the RNG in ``states[index]`` by 2**64 steps.

    :type states: 1D array, dtype=xoroshiro128p_dtype
    :param states: array of RNG states
    :type index: int64
    :param index: offset in states to update
    """
    index = int64(index)
    jump = (uint64(13739361407582206667), uint64(15594563132006766882))
    s0 = uint64(0)
    s1 = uint64(0)
    for i in range(2):
        for b in range(64):
            if jump[i] & uint64(1) << uint32(b):
                s0 ^= states[index]['s0']
                s1 ^= states[index]['s1']
            xoroshiro128p_next(states, index)
    states[index]['s0'] = s0
    states[index]['s1'] = s1