import functools
import numpy as np
import scipy.fft as fft
from .._shared.utils import _supported_float_type
def _get_nd_butterworth_filter(shape, factor, order, high_pass, real, dtype=np.float64, squared_butterworth=True):
    """Create a N-dimensional Butterworth mask for an FFT

    Parameters
    ----------
    shape : tuple of int
        Shape of the n-dimensional FFT and mask.
    factor : float
        Fraction of mask dimensions where the cutoff should be.
    order : float
        Controls the slope in the cutoff region.
    high_pass : bool
        Whether the filter is high pass (low frequencies attenuated) or
        low pass (high frequencies are attenuated).
    real : bool
        Whether the FFT is of a real (True) or complex (False) image
    squared_butterworth : bool, optional
        When True, the square of the Butterworth filter is used.

    Returns
    -------
    wfilt : ndarray
        The FFT mask.

    """
    ranges = []
    for i, d in enumerate(shape):
        axis = np.arange(-(d - 1) // 2, (d - 1) // 2 + 1) / (d * factor)
        ranges.append(fft.ifftshift(axis ** 2))
    if real:
        limit = d // 2 + 1
        ranges[-1] = ranges[-1][:limit]
    q2 = functools.reduce(np.add, np.meshgrid(*ranges, indexing='ij', sparse=True))
    q2 = q2.astype(dtype)
    q2 = np.power(q2, order)
    wfilt = 1 / (1 + q2)
    if high_pass:
        wfilt *= q2
    if not squared_butterworth:
        np.sqrt(wfilt, out=wfilt)
    return wfilt