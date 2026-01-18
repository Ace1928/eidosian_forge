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
def test_slice_spacing():
    padded = np.pad(SAMPLE, ((2, 4), (5, 2)), mode='constant')
    nrow, ncol = SAMPLE.shape
    result = regionprops(padded)[0].slice
    expected = (slice(2, 2 + nrow), slice(5, 5 + ncol))
    spacing = (2, 0.2)
    result = regionprops(padded, spacing=spacing)[0].slice
    assert_equal(result, expected)