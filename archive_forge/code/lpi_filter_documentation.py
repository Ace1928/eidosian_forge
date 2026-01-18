import numpy as np
import scipy.fft as fft
from .._shared.utils import _supported_float_type, check_nD
Apply the filter to the given data.

        Parameters
        ----------
        data : (M, N) ndarray

        