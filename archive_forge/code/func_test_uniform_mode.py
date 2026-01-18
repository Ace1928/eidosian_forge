import pytest
import copy
import numpy as np
from skimage._shared.testing import assert_array_equal
from skimage import data
from skimage.feature import BRIEF, corner_peaks, corner_harris
from skimage._shared import testing
@pytest.mark.parametrize('dtype', ['float32', 'float64', 'uint8', 'int'])
def test_uniform_mode(dtype):
    """Verify the computed BRIEF descriptors with expected for uniform mode."""
    img = data.coins().astype(dtype)
    keypoints = corner_peaks(corner_harris(img), min_distance=5, threshold_abs=0, threshold_rel=0.1)
    extractor = BRIEF(descriptor_size=8, sigma=2, mode='uniform', rng=1)
    BRIEF(descriptor_size=8, sigma=2, mode='uniform', rng=1)
    extractor.extract(img, keypoints[:8])
    expected = np.array([[0, 1, 0, 1, 0, 1, 1, 0], [0, 1, 0, 0, 0, 1, 0, 1], [0, 1, 0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 1, 0, 1, 1], [0, 0, 1, 0, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 0, 0, 1, 1, 1], [1, 0, 1, 1, 1, 0, 0, 1]], dtype=bool)
    assert_array_equal(extractor.descriptors, expected)