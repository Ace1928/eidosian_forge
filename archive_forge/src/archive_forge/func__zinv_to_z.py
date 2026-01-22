import warnings
import copy
from math import sqrt
import cupy
from cupyx.scipy import linalg
from cupyx.scipy.interpolate import make_interp_spline
from cupyx.scipy.linalg import expm, block_diag
from cupyx.scipy.signal._lti_conversion import (
from cupyx.scipy.signal._iir_filter_conversions import (
from cupyx.scipy.signal._filter_design import (
@staticmethod
def _zinv_to_z(num, den):
    """Change a transfer function from the variable `z` to `z**-1`.

        Parameters
        ----------
        num, den: 1d array_like
            Sequences representing the coefficients of the numerator and
            denominator polynomials, in order of ascending degree of 'z**-1'.
            That is, ``5 + 3 z**-1 + 2 z**-2`` is presented as ``[5, 3, 2]``.

        Returns
        -------
        num, den: 1d array_like
            Sequences representing the coefficients of the numerator and
            denominator polynomials, in order of descending degree of 'z'.
            That is, ``5z**2 + 3z + 2`` is presented as ``[5, 3, 2]``.
        """
    diff = len(num) - len(den)
    if diff > 0:
        den = cupy.hstack((den, cupy.zeros(diff)))
    elif diff < 0:
        num = cupy.hstack((num, cupy.zeros(-diff)))
    return (num, den)