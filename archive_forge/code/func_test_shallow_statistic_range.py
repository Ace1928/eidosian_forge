import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
@pytest.mark.parametrize('mode', ['maximum', 'mean', 'median', 'minimum'])
def test_shallow_statistic_range(self, mode):
    test = np.arange(120).reshape(4, 5, 6)
    pad_amt = [(1, 1) for _ in test.shape]
    assert_array_equal(np.pad(test, pad_amt, mode='edge'), np.pad(test, pad_amt, mode=mode, stat_length=1))