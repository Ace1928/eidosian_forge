import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
@pytest.mark.parametrize('mode', _all_modes.keys())
def test_pad_non_empty_dimension(self, mode):
    result = np.pad(np.ones((2, 0, 2)), ((3,), (0,), (1,)), mode=mode)
    assert result.shape == (8, 0, 4)