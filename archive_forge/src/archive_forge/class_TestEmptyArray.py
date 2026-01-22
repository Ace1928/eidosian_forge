import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
class TestEmptyArray:
    """Check how padding behaves on arrays with an empty dimension."""

    @pytest.mark.parametrize('mode', sorted(_all_modes.keys() - {'constant', 'empty'}))
    def test_pad_empty_dimension(self, mode):
        match = "can't extend empty axis 0 using modes other than 'constant' or 'empty'"
        with pytest.raises(ValueError, match=match):
            np.pad([], 4, mode=mode)
        with pytest.raises(ValueError, match=match):
            np.pad(np.ndarray(0), 4, mode=mode)
        with pytest.raises(ValueError, match=match):
            np.pad(np.zeros((0, 3)), ((1,), (0,)), mode=mode)

    @pytest.mark.parametrize('mode', _all_modes.keys())
    def test_pad_non_empty_dimension(self, mode):
        result = np.pad(np.ones((2, 0, 2)), ((3,), (0,), (1,)), mode=mode)
        assert result.shape == (8, 0, 4)