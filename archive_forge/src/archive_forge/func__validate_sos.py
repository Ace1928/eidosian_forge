import operator
from math import pi
import warnings
import cupy
from cupy.polynomial.polynomial import (
import cupyx.scipy.fft as sp_fft
from cupyx import jit
from cupyx.scipy._lib._util import float_factorial
from cupyx.scipy.signal._polyutils import roots
def _validate_sos(sos):
    """Helper to validate a SOS input"""
    sos = cupy.atleast_2d(sos)
    if sos.ndim != 2:
        raise ValueError('sos array must be 2D')
    n_sections, m = sos.shape
    if m != 6:
        raise ValueError('sos array must be shape (n_sections, 6)')
    if (sos[:, 3] - 1 > 1e-15).any():
        raise ValueError('sos[:, 3] should be all ones')
    return (sos, n_sections)