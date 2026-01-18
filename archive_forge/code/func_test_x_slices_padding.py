import math
from itertools import product
from typing import cast, get_args, Literal
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from scipy.fft import fftshift
from scipy.stats import norm as normal_distribution  # type: ignore
from scipy.signal import get_window, welch, stft, istft, spectrogram
from scipy.signal._short_time_fft import FFT_MODE_TYPE, \
from scipy.signal.windows import gaussian
def test_x_slices_padding():
    """Verify padding.

    The reference arrays were taken from  the docstrings of `zero_ext`,
    `const_ext`, `odd_ext()`, and `even_ext()` from the _array_tools module.
    """
    SFT = ShortTimeFFT(np.ones(5), hop=4, fs=1)
    x = np.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]], dtype=float)
    d = {'zeros': [[[0, 0, 1, 2, 3], [0, 0, 0, 1, 4]], [[3, 4, 5, 0, 0], [4, 9, 16, 0, 0]]], 'edge': [[[1, 1, 1, 2, 3], [0, 0, 0, 1, 4]], [[3, 4, 5, 5, 5], [4, 9, 16, 16, 16]]], 'even': [[[3, 2, 1, 2, 3], [4, 1, 0, 1, 4]], [[3, 4, 5, 4, 3], [4, 9, 16, 9, 4]]], 'odd': [[[-1, 0, 1, 2, 3], [-4, -1, 0, 1, 4]], [[3, 4, 5, 6, 7], [4, 9, 16, 23, 28]]]}
    for p_, xx in d.items():
        gen = SFT._x_slices(np.array(x), 0, 0, 2, padding=cast(PAD_TYPE, p_))
        yy = np.array([y_.copy() for y_ in gen])
        assert_equal(yy, xx, err_msg=f"Failed '{p_}' padding.")