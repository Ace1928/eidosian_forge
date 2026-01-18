from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage import data, filters, img_as_float
from skimage._shared.testing import run_in_parallel, expected_warnings
from skimage.segmentation import slic
def test_raises_ValueError_if_input_has_NaN():
    img = np.zeros((4, 5), dtype=float)
    img[2, 3] = np.nan
    with pytest.raises(ValueError):
        slic(img, channel_axis=None)
    mask = ~np.isnan(img)
    slic(img, mask=mask, channel_axis=None)