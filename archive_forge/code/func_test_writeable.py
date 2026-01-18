import numpy as np
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.lib.stride_tricks import (
import pytest
def test_writeable(self):
    arr = np.arange(5)
    view = sliding_window_view(arr, 2, writeable=False)
    assert_(not view.flags.writeable)
    with pytest.raises(ValueError, match='assignment destination is read-only'):
        view[0, 0] = 3
    view = sliding_window_view(arr, 2, writeable=True)
    assert_(view.flags.writeable)
    view[0, 1] = 3
    assert_array_equal(arr, np.array([0, 3, 2, 3, 4]))