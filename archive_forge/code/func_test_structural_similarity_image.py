import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.metrics import structural_similarity
def test_structural_similarity_image():
    N = 100
    X = (np.random.rand(N, N) * 255).astype(np.uint8)
    Y = (np.random.rand(N, N) * 255).astype(np.uint8)
    S0 = structural_similarity(X, X, win_size=3)
    assert_equal(S0, 1)
    S1 = structural_similarity(X, Y, win_size=3)
    assert S1 < 0.3
    S2 = structural_similarity(X, Y, win_size=11, gaussian_weights=True)
    assert S2 < 0.3
    mssim0, S3 = structural_similarity(X, Y, full=True)
    assert_equal(S3.shape, X.shape)
    mssim = structural_similarity(X, Y)
    assert_equal(mssim0, mssim)
    assert_equal(structural_similarity(X, X), 1.0)