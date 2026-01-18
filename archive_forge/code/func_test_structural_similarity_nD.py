import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.metrics import structural_similarity
@pytest.mark.parametrize('dtype', [np.uint8, np.float32, np.float64])
def test_structural_similarity_nD(dtype):
    N = 10
    for ndim in range(1, 5):
        xsize = [N] * 5
        X = (np.random.rand(*xsize) * 255).astype(dtype)
        Y = (np.random.rand(*xsize) * 255).astype(dtype)
        mssim = structural_similarity(X, Y, win_size=3, data_range=255.0)
        assert mssim.dtype == np.float64
        assert mssim < 0.05