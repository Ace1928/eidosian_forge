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
@pytest.mark.parametrize('spacing', [np.nan, np.inf, -np.inf])
def test_spacing_parameter_nan_inf(spacing):
    """Test the _normalize_spacing code."""
    with pytest.raises(ValueError):
        regionprops(SAMPLE, spacing=spacing)[0].centroid