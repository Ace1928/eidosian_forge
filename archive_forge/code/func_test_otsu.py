import inspect
import numpy as np
import pytest
from skimage import data, morphology, util
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.filters import rank
from skimage.filters.rank import __all__ as all_rank_filters
from skimage.filters.rank import __3Dfilters as _3d_rank_filters
from skimage.filters.rank import subtract_mean
from skimage.morphology import ball, disk, gray
from skimage.util import img_as_float, img_as_ubyte
def test_otsu(self):
    test = np.tile([128, 145, 103, 127, 165, 83, 127, 185, 63, 127, 205, 43, 127, 225, 23, 127], (16, 1))
    test = test.astype(np.uint8)
    res = np.tile([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1], (16, 1))
    footprint = np.ones((6, 6), dtype=np.uint8)
    th = 1 * (test >= rank.otsu(test, footprint))
    assert_equal(th, res)