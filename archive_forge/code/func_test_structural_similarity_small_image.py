import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.metrics import structural_similarity
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_structural_similarity_small_image(dtype):
    X = np.zeros((5, 5), dtype=dtype)
    assert_equal(structural_similarity(X, X, win_size=3, data_range=1.0), 1.0)
    assert_equal(structural_similarity(X, X, win_size=5, data_range=1.0), 1.0)
    with pytest.raises(ValueError):
        structural_similarity(X, X)