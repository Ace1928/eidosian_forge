import operator
import math
from math import prod as _prod
import timeit
import warnings
from scipy.spatial import cKDTree
from . import _sigtools
from ._ltisys import dlti
from ._upfirdn import upfirdn, _output_len, _upfirdn_modes
from scipy import linalg, fft as sp_fft
from scipy import ndimage
from scipy.fft._helper import _init_nd_shape_and_axes
import numpy as np
from scipy.special import lambertw
from .windows import get_window
from ._arraytools import axis_slice, axis_reverse, odd_ext, even_ext, const_ext
from ._filter_design import cheby1, _validate_sos, zpk2sos
from ._fir_filter_design import firwin
from ._sosfilt import _sosfilt
def _conv_ops(x_shape, h_shape, mode):
    """
    Find the number of operations required for direct/fft methods of
    convolution. The direct operations were recorded by making a dummy class to
    record the number of operations by overriding ``__mul__`` and ``__add__``.
    The FFT operations rely on the (well-known) computational complexity of the
    FFT (and the implementation of ``_freq_domain_conv``).

    """
    if mode == 'full':
        out_shape = [n + k - 1 for n, k in zip(x_shape, h_shape)]
    elif mode == 'valid':
        out_shape = [abs(n - k) + 1 for n, k in zip(x_shape, h_shape)]
    elif mode == 'same':
        out_shape = x_shape
    else:
        raise ValueError(f"Acceptable mode flags are 'valid', 'same', or 'full', not mode={mode}")
    s1, s2 = (x_shape, h_shape)
    if len(x_shape) == 1:
        s1, s2 = (s1[0], s2[0])
        if mode == 'full':
            direct_ops = s1 * s2
        elif mode == 'valid':
            direct_ops = (s2 - s1 + 1) * s1 if s2 >= s1 else (s1 - s2 + 1) * s2
        elif mode == 'same':
            direct_ops = s1 * s2 if s1 < s2 else s1 * s2 - s2 // 2 * ((s2 + 1) // 2)
    elif mode == 'full':
        direct_ops = min(_prod(s1), _prod(s2)) * _prod(out_shape)
    elif mode == 'valid':
        direct_ops = min(_prod(s1), _prod(s2)) * _prod(out_shape)
    elif mode == 'same':
        direct_ops = _prod(s1) * _prod(s2)
    full_out_shape = [n + k - 1 for n, k in zip(x_shape, h_shape)]
    N = _prod(full_out_shape)
    fft_ops = 3 * N * np.log(N)
    return (fft_ops, direct_ops)