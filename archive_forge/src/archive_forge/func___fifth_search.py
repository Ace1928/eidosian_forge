import re
import numpy as np
from numba import jit
from .intervals import INTERVALS
from .._cache import cache
from ..util.exceptions import ParameterError
from typing import Dict, List, Union, overload
from ..util.decorators import vectorize
from .._typing import _ScalarOrSequence, _FloatLike_co, _SequenceLike
@jit(nopython=True, nogil=True, cache=True)
def __fifth_search(interval, tolerance):
    """Accelerated helper function for finding the number of fifths
    to get within tolerance of a given interval.

    This implementation will give up after 32 fifths
    """
    log_tolerance = np.abs(np.log2(tolerance))
    for power in range(32):
        for sign in [1, -1]:
            if np.abs(np.log2(__bo_fold(interval / 3.0 ** (power * sign)))) <= log_tolerance:
                return power * sign
        power += 1
    return power