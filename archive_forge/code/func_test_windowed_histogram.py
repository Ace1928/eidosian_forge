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
def test_windowed_histogram(self):
    image8 = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]], dtype=np.uint8)
    elem = np.ones((3, 3), dtype=np.uint8)
    outf = np.empty(image8.shape + (2,), dtype=float)
    mask = np.ones(image8.shape, dtype=np.uint8)
    pop = np.array([[4, 6, 6, 6, 4], [6, 9, 9, 9, 6], [6, 9, 9, 9, 6], [6, 9, 9, 9, 6], [4, 6, 6, 6, 4]], dtype=float)
    r0 = np.array([[3, 4, 3, 4, 3], [4, 5, 3, 5, 4], [3, 3, 0, 3, 3], [4, 5, 3, 5, 4], [3, 4, 3, 4, 3]], dtype=float) / pop
    r1 = np.array([[1, 2, 3, 2, 1], [2, 4, 6, 4, 2], [3, 6, 9, 6, 3], [2, 4, 6, 4, 2], [1, 2, 3, 2, 1]], dtype=float) / pop
    rank.windowed_histogram(image=image8, footprint=elem, out=outf, mask=mask)
    assert_equal(r0, outf[:, :, 0])
    assert_equal(r1, outf[:, :, 1])
    larger_output = rank.windowed_histogram(image=image8, footprint=elem, mask=mask, n_bins=5)
    assert larger_output.shape[2] == 5