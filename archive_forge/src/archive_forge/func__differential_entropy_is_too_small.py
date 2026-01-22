from __future__ import annotations
import math
import numpy as np
from scipy import special
from ._axis_nan_policy import _axis_nan_policy_factory, _broadcast_arrays
def _differential_entropy_is_too_small(samples, kwargs, axis=-1):
    values = samples[0]
    n = values.shape[axis]
    window_length = kwargs.get('window_length', math.floor(math.sqrt(n) + 0.5))
    if not 2 <= 2 * window_length < n:
        return True
    return False