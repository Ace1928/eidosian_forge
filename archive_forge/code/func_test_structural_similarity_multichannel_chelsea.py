import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.metrics import structural_similarity
def test_structural_similarity_multichannel_chelsea():
    Xc = data.chelsea()
    sigma = 15.0
    Yc = np.clip(Xc + sigma * np.random.randn(*Xc.shape), 0, 255)
    Yc = Yc.astype(Xc.dtype)
    mssim = structural_similarity(Xc, Yc, channel_axis=-1)
    mssim_sep = [structural_similarity(Yc[..., c], Xc[..., c]) for c in range(Xc.shape[-1])]
    assert_almost_equal(mssim, np.mean(mssim_sep))
    assert_equal(structural_similarity(Xc, Xc, channel_axis=-1), 1.0)