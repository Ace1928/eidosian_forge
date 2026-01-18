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
def test_population(self):
    image = np.zeros((5, 5), dtype=np.uint8)
    elem = np.ones((3, 3), dtype=np.uint8)
    out = np.empty_like(image)
    mask = np.ones(image.shape, dtype=np.uint8)
    rank.pop(image=image, footprint=elem, out=out, mask=mask)
    r = np.array([[4, 6, 6, 6, 4], [6, 9, 9, 9, 6], [6, 9, 9, 9, 6], [6, 9, 9, 9, 6], [4, 6, 6, 6, 4]])
    assert_equal(r, out)