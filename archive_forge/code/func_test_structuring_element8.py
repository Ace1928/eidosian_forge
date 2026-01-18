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
def test_structuring_element8(self):
    r = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 255, 0, 0, 0], [0, 0, 255, 255, 255, 0], [0, 0, 0, 255, 255, 0], [0, 0, 0, 0, 0, 0]])
    image = np.zeros((6, 6), dtype=np.uint8)
    image[2, 2] = 255
    elem = np.asarray([[1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=np.uint8)
    out = np.empty_like(image)
    mask = np.ones(image.shape, dtype=np.uint8)
    rank.maximum(image=image, footprint=elem, out=out, mask=mask, shift_x=1, shift_y=1)
    assert_equal(r, out)
    image = np.zeros((6, 6), dtype=np.uint16)
    image[2, 2] = 255
    out = np.empty_like(image)
    rank.maximum(image=image, footprint=elem, out=out, mask=mask, shift_x=1, shift_y=1)
    assert_equal(r, out)