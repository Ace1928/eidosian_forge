import numpy as np
from numpy.testing import (
import pytest
def test_zero_poly_dtype(self):
    """
        Regression test for gh-16354.
        """
    z = np.array([0, 0, 0])
    p = np.poly1d(z.astype(np.int64))
    assert_equal(p.coeffs.dtype, np.int64)
    p = np.poly1d(z.astype(np.float32))
    assert_equal(p.coeffs.dtype, np.float32)
    p = np.poly1d(z.astype(np.complex64))
    assert_equal(p.coeffs.dtype, np.complex64)