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
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_subtract_mean_underflow_correction(dtype):
    footprint = np.ones((1, 3))
    arr = np.array([[10, 10, 10]], dtype=dtype)
    result = subtract_mean(arr, footprint)
    if dtype == np.uint8:
        expected_val = 127
    else:
        expected_val = (arr.max() + 1) // 2 - 1
    assert np.all(result == expected_val)