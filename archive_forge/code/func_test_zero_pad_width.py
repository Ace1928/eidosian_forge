import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
@pytest.mark.parametrize('pad_width', [0, (0, 0), ((0, 0), (0, 0))])
@pytest.mark.parametrize('mode', _all_modes.keys())
def test_zero_pad_width(self, pad_width, mode):
    arr = np.arange(30).reshape(6, 5)
    assert_array_equal(arr, np.pad(arr, pad_width, mode=mode))