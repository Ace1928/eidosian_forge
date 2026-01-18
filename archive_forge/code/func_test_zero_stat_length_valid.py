import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
@pytest.mark.filterwarnings('ignore:Mean of empty slice:RuntimeWarning')
@pytest.mark.filterwarnings('ignore:invalid value encountered in( scalar)? divide:RuntimeWarning')
@pytest.mark.parametrize('mode', ['mean', 'median'])
def test_zero_stat_length_valid(self, mode):
    arr = np.pad([1.0, 2.0], (1, 2), mode, stat_length=0)
    expected = np.array([np.nan, 1.0, 2.0, np.nan, np.nan])
    assert_equal(arr, expected)