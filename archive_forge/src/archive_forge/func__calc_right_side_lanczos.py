import operator
import warnings
import numpy as np
from scipy import linalg, special, fft as sp_fft
def _calc_right_side_lanczos(n, m):
    return np.sinc(2.0 * np.arange(n, m) / (m - 1) - 1.0)