import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
@pytest.mark.parametrize('axis', [0, -1])
def test_unique_1d_with_axis(self, axis):
    x = np.array([4, 3, 2, 3, 2, 1, 2, 2])
    uniq = unique(x, axis=axis)
    assert_array_equal(uniq, [1, 2, 3, 4])