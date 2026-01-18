import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
def test_repeated_wrapping(self):
    """
        Check wrapping on each side individually if the wrapped area is longer
        than the original array.
        """
    a = np.arange(5)
    b = np.pad(a, (12, 0), mode='wrap')
    assert_array_equal(np.r_[a, a, a, a][3:], b)
    a = np.arange(5)
    b = np.pad(a, (0, 12), mode='wrap')
    assert_array_equal(np.r_[a, a, a, a][:-3], b)