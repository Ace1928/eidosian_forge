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
@pytest.mark.skipif(Version(np.__version__) < Version('1.25'), reason='Older NumPy throws a few extra warnings here')
@pytest.mark.parametrize('in_range,out_range', [('image', 'dtype'), ('dtype', 'image')])
def test_rescale_nan_warning(in_range, out_range):
    image = np.arange(12, dtype=float).reshape(3, 4)
    image[1, 1] = np.nan
    with expected_warnings(['One or more intensity levels are NaN\\. Rescaling will broadcast NaN to the full image\\.']):
        exposure.rescale_intensity(image, in_range, out_range)