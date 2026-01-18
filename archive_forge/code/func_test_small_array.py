import math
import unittest
import numpy as np
from numpy.testing import assert_equal
from pytest import raises, warns
from skimage._shared.testing import expected_warnings
from skimage.morphology import extrema
def test_small_array(self):
    """Test output for arrays with dimension smaller 3.

        If any dimension of an array is smaller than 3 and `allow_borders` is
        false a footprint, which has at least 3 elements in each
        dimension, can't be applied. This is an implementation detail so
        `local_maxima` should still return valid output (see gh-3261).

        If `allow_borders` is true the array is padded internally and there is
        no problem.
        """
    warning_msg = "maxima can't exist .* any dimension smaller 3 .*"
    x = np.array([0, 1])
    extrema.local_maxima(x, allow_borders=True)
    with warns(UserWarning, match=warning_msg):
        result = extrema.local_maxima(x, allow_borders=False)
    assert_equal(result, [0, 0])
    assert result.dtype == bool
    x = np.array([[1, 2], [2, 2]])
    extrema.local_maxima(x, allow_borders=True, indices=True)
    with warns(UserWarning, match=warning_msg):
        result = extrema.local_maxima(x, allow_borders=False, indices=True)
    assert_equal(result, np.zeros((2, 0), dtype=np.intp))
    assert result[0].dtype == np.intp
    assert result[1].dtype == np.intp