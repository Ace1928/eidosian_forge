import numpy as np
from numba import uint64, uint32, uint16, uint8
from numba.core.extending import register_jitable
from numba.np.random._constants import (UINT32_MAX, UINT64_MAX,
from numba.np.random.generator_core import next_uint32, next_uint64

    Check that low and high are within the bounds
    for the given datatype.
    