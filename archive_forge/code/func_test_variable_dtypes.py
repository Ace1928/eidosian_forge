import math
import numpy as np
import pytest
from numpy.testing import (
from scipy import ndimage as ndi
from skimage import data, util
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.draw import disk
from skimage.exposure import histogram
from skimage.filters._multiotsu import (
from skimage.filters.thresholding import (
@pytest.mark.parametrize('threshold_func', [threshold_local, threshold_niblack, threshold_sauvola])
@pytest.mark.parametrize('dtype', [np.uint8, np.int16, np.float16, np.float32])
def test_variable_dtypes(threshold_func, dtype):
    r = 255 * np.random.rand(32, 16)
    r = r.astype(dtype, copy=False)
    kwargs = {}
    if threshold_func is threshold_local:
        kwargs = dict(block_size=9)
    elif threshold_func is threshold_sauvola:
        kwargs = dict(r=128)
    expected = threshold_func(r.astype(float), **kwargs)
    out = threshold_func(r, **kwargs)
    assert out.dtype == _supported_float_type(dtype)
    assert_allclose(out, expected, rtol=1e-05, atol=1e-05)