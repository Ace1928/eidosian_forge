import operator
from math import gcd
import cupy
from cupyx.scipy.fft import fft, rfft, fftfreq, ifft, irfft, ifftshift
from cupyx.scipy.signal._iir_filter_design import cheby1
from cupyx.scipy.signal._fir_filter_design import firwin
from cupyx.scipy.signal._iir_filter_conversions import zpk2sos
from cupyx.scipy.signal._ltisys import dlti
from cupyx.scipy.signal._upfirdn import upfirdn, _output_len
from cupyx.scipy.signal._signaltools import (
from cupyx.scipy.signal.windows._windows import get_window
def _design_resample_poly(up, down, window):
    """
    Design a prototype FIR low-pass filter using the window method
    for use in polyphase rational resampling.

    Parameters
    ----------
    up : int
        The upsampling factor.
    down : int
        The downsampling factor.
    window : string or tuple
        Desired window to use to design the low-pass filter.
        See below for details.

    Returns
    -------
    h : array
        The computed FIR filter coefficients.

    See Also
    --------
    resample_poly : Resample up or down using the polyphase method.

    Notes
    -----
    The argument `window` specifies the FIR low-pass filter design.
    The functions `cusignal.get_window` and `cusignal.firwin`
    are called to generate the appropriate filter coefficients.

    The returned array of coefficients will always be of data type
    `complex128` to maintain precision. For use in lower-precision
    filter operations, this array should be converted to the desired
    data type before providing it to `cusignal.resample_poly`.

    """
    g_ = gcd(up, down)
    up //= g_
    down //= g_
    max_rate = max(up, down)
    f_c = 1.0 / max_rate
    half_len = 10 * max_rate
    h = firwin(2 * half_len + 1, f_c, window=window)
    return h