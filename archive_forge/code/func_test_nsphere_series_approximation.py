import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage._shared.testing import fetch
from skimage.morphology import footprints
@pytest.mark.parametrize('function', ['disk', 'ball'])
@pytest.mark.parametrize('radius', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 75, 100])
def test_nsphere_series_approximation(function, radius):
    fp_func = getattr(footprints, function)
    expected = fp_func(radius, strict_radius=False, decomposition=None)
    footprint_sequence = fp_func(radius, strict_radius=False, decomposition='sequence')
    approximate = footprints.footprint_from_sequence(footprint_sequence)
    assert approximate.shape == expected.shape
    error = np.sum(np.abs(expected.astype(int) - approximate.astype(int)))
    if radius == 1:
        assert error == 0
    else:
        max_error = 0.1 if function == 'disk' else 0.15
        assert error / expected.size <= max_error