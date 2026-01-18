import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.metrics import structural_similarity
def test_structural_similarity_patch_range():
    N = 51
    X = (np.random.rand(N, N) * 255).astype(np.uint8)
    Y = (np.random.rand(N, N) * 255).astype(np.uint8)
    assert structural_similarity(X, Y, win_size=N) < 0.1
    assert_equal(structural_similarity(X, X, win_size=N), 1)