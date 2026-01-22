import numpy as np
import scipy.fft as fft
from .._shared.utils import _supported_float_type, check_nD
def _center(x, oshape):
    """Return an array of shape ``oshape`` from the center of array ``x``."""
    start = (np.array(x.shape) - np.array(oshape)) // 2
    out = x[tuple((slice(s, s + n) for s, n in zip(start, oshape)))]
    return out