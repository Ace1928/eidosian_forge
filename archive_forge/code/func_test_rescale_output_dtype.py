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
@pytest.mark.parametrize('out_range, out_dtype', [('uint8', np.uint8), ('uint10', np.uint16), ('uint12', np.uint16), ('uint16', np.uint16), ('float', float)])
def test_rescale_output_dtype(out_range, out_dtype):
    image = np.array([-128, 0, 127], dtype=np.int8)
    output_image = exposure.rescale_intensity(image, out_range=out_range)
    assert output_image.dtype == out_dtype