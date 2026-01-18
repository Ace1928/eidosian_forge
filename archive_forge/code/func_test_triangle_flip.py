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
def test_triangle_flip():
    img = data.camera()
    inv_img = np.invert(img)
    t = threshold_triangle(inv_img)
    t_inv_img = inv_img > t
    t_inv_inv_img = np.invert(t_inv_img)
    t = threshold_triangle(img)
    t_img = img > t
    unequal_pos = np.where(t_img.ravel() != t_inv_inv_img.ravel())
    assert len(unequal_pos[0]) / t_img.size < 0.01