import operator
import numpy as np
from numpy import ndarray, float_
import numpy.core.umath as umath
import numpy.testing
from numpy.testing import (
from .core import mask_or, getmask, masked_array, nomask, masked, filled
from unittest import TestCase
def approx(a, b, fill_value=True, rtol=1e-05, atol=1e-08):
    """
    Returns true if all components of a and b are equal to given tolerances.

    If fill_value is True, masked values considered equal. Otherwise,
    masked values are considered unequal.  The relative error rtol should
    be positive and << 1.0 The absolute error atol comes into play for
    those elements of b that are very small or zero; it says how small a
    must be also.

    """
    m = mask_or(getmask(a), getmask(b))
    d1 = filled(a)
    d2 = filled(b)
    if d1.dtype.char == 'O' or d2.dtype.char == 'O':
        return np.equal(d1, d2).ravel()
    x = filled(masked_array(d1, copy=False, mask=m), fill_value).astype(float_)
    y = filled(masked_array(d2, copy=False, mask=m), 1).astype(float_)
    d = np.less_equal(umath.absolute(x - y), atol + rtol * umath.absolute(y))
    return d.ravel()