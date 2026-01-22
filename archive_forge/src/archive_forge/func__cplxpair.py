import math
import operator
import warnings
import numpy
import numpy as np
from numpy import (atleast_1d, poly, polyval, roots, real, asarray,
from numpy.polynomial.polynomial import polyval as npp_polyval
from numpy.polynomial.polynomial import polyvalfromroots
from scipy import special, optimize, fft as sp_fft
from scipy.special import comb
from scipy._lib._util import float_factorial
def _cplxpair(z, tol=None):
    """
    Sort into pairs of complex conjugates.

    Complex conjugates in `z` are sorted by increasing real part. In each
    pair, the number with negative imaginary part appears first.

    If pairs have identical real parts, they are sorted by increasing
    imaginary magnitude.

    Two complex numbers are considered a conjugate pair if their real and
    imaginary parts differ in magnitude by less than ``tol * abs(z)``.  The
    pairs are forced to be exact complex conjugates by averaging the positive
    and negative values.

    Purely real numbers are also sorted, but placed after the complex
    conjugate pairs. A number is considered real if its imaginary part is
    smaller than `tol` times the magnitude of the number.

    Parameters
    ----------
    z : array_like
        1-D input array to be sorted.
    tol : float, optional
        Relative tolerance for testing realness and conjugate equality.
        Default is ``100 * spacing(1)`` of `z`'s data type (i.e., 2e-14 for
        float64)

    Returns
    -------
    y : ndarray
        Complex conjugate pairs followed by real numbers.

    Raises
    ------
    ValueError
        If there are any complex numbers in `z` for which a conjugate
        cannot be found.

    See Also
    --------
    _cplxreal

    Examples
    --------
    >>> from scipy.signal._filter_design import _cplxpair
    >>> a = [4, 3, 1, 2-2j, 2+2j, 2-1j, 2+1j, 2-1j, 2+1j, 1+1j, 1-1j]
    >>> z = _cplxpair(a)
    >>> print(z)
    [ 1.-1.j  1.+1.j  2.-1.j  2.+1.j  2.-1.j  2.+1.j  2.-2.j  2.+2.j  1.+0.j
      3.+0.j  4.+0.j]
    """
    z = atleast_1d(z)
    if z.size == 0 or np.isrealobj(z):
        return np.sort(z)
    if z.ndim != 1:
        raise ValueError('z must be 1-D')
    zc, zr = _cplxreal(z, tol)
    zc = np.dstack((zc.conj(), zc)).flatten()
    z = np.append(zc, zr)
    return z