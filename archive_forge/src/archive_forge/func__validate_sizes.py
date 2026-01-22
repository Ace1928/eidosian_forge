import cmath
import numbers
import cupy
from numpy import pi
from cupyx.scipy.fft import fft, ifft, next_fast_len
def _validate_sizes(n, m):
    if n < 1 or not isinstance(n, numbers.Integral):
        raise ValueError(f'Invalid number of CZT data points ({n}) specified. n must be positive and integer type.')
    if m is None:
        m = n
    elif m < 1 or not isinstance(m, numbers.Integral):
        raise ValueError(f'Invalid number of CZT output points ({m}) specified. m must be positive and integer type.')
    return m