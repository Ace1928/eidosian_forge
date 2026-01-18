import warnings
import numpy as np
import pytest
from numpy.testing import (
from packaging.version import Version
from skimage import data
from skimage import exposure
from skimage import util
from skimage.color import rgb2gray
from skimage.exposure.exposure import intensity_range
from skimage.util.dtype import dtype_range
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
@pytest.mark.parametrize('dtype', [np.int8, np.float32])
def test_wrong_source_range(dtype):
    im = np.array([-1, 100], dtype=dtype)
    with pytest.raises(ValueError, match='Incorrect value for `source_range` argument'):
        frequencies, bin_centers = exposure.histogram(im, source_range='foobar')