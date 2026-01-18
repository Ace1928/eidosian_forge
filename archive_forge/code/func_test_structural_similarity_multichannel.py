import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.metrics import structural_similarity
@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1])
def test_structural_similarity_multichannel(channel_axis):
    N = 100
    X = (np.random.rand(N, N) * 255).astype(np.uint8)
    Y = (np.random.rand(N, N) * 255).astype(np.uint8)
    S1 = structural_similarity(X, Y, win_size=3)
    Xc = np.tile(X[..., np.newaxis], (1, 1, 3))
    Yc = np.tile(Y[..., np.newaxis], (1, 1, 3))
    Xc, Yc = (np.moveaxis(_arr, -1, channel_axis) for _arr in (Xc, Yc))
    S2 = structural_similarity(Xc, Yc, channel_axis=channel_axis, win_size=3)
    assert_almost_equal(S1, S2)
    m, S3 = structural_similarity(Xc, Yc, channel_axis=channel_axis, full=True)
    assert_equal(S3.shape, Xc.shape)
    m, grad = structural_similarity(Xc, Yc, channel_axis=channel_axis, gradient=True)
    assert_equal(grad.shape, Xc.shape)
    m, grad, S3 = structural_similarity(Xc, Yc, channel_axis=channel_axis, full=True, gradient=True)
    assert_equal(grad.shape, Xc.shape)
    assert_equal(S3.shape, Xc.shape)
    with pytest.raises(ValueError):
        structural_similarity(Xc, Yc, win_size=7, channel_axis=None)