import math
import re
import numpy as np
import pytest
import scipy.ndimage as ndi
from numpy.testing import (
from skimage import data, draw, transform
from skimage._shared import testing
from skimage.measure._regionprops import (
from skimage.segmentation import slic
@pytest.mark.parametrize('spacing', [[2.1, 2.2, 2.3], [2.0, 2.0, 2.0], [2, 2, 2]])
def test_spacing_parameter_3d(spacing):
    """Test the _normalize_spacing code."""
    Mpqr = get_moment3D_function(SAMPLE_3D, spacing=spacing)
    cZ = Mpqr(1, 0, 0) / Mpqr(0, 0, 0)
    cY = Mpqr(0, 1, 0) / Mpqr(0, 0, 0)
    cX = Mpqr(0, 0, 1) / Mpqr(0, 0, 0)
    centroid = regionprops(SAMPLE_3D, spacing=spacing)[0].centroid
    assert_array_almost_equal(centroid, (cZ, cY, cX))