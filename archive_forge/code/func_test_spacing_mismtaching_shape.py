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
@pytest.mark.parametrize('spacing', ([1], [[1, 1]], (1, 1, 1)))
def test_spacing_mismtaching_shape(spacing):
    with pytest.raises(ValueError, match="spacing isn't a scalar nor a sequence"):
        regionprops(SAMPLE, spacing=spacing)[0].centroid