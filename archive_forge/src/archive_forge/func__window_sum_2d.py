import math
import numpy as np
from scipy.signal import fftconvolve
from .._shared.utils import check_nD, _supported_float_type
def _window_sum_2d(image, window_shape):
    window_sum = np.cumsum(image, axis=0)
    window_sum = window_sum[window_shape[0]:-1] - window_sum[:-window_shape[0] - 1]
    window_sum = np.cumsum(window_sum, axis=1)
    window_sum = window_sum[:, window_shape[1]:-1] - window_sum[:, :-window_shape[1] - 1]
    return window_sum