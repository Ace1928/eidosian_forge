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
def test_triangle_float_images():
    text = data.text()
    int_bins = text.max() - text.min() + 1
    assert round(threshold_triangle(text.astype(float), nbins=int_bins)) == 104
    assert round(threshold_triangle(text / 255.0, nbins=int_bins) * 255) == 104
    assert round(threshold_triangle(np.invert(text).astype(float), nbins=int_bins)) == 151
    assert round(threshold_triangle(np.invert(text) / 255.0, nbins=int_bins) * 255) == 151