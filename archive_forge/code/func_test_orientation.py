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
def test_orientation():
    orient = regionprops(SAMPLE)[0].orientation
    target_orient = -1.4663278802756865
    assert_almost_equal(orient, target_orient)
    orient = regionprops(SAMPLE, spacing=(2, 2))[0].orientation
    assert_almost_equal(orient, target_orient)
    diag = np.eye(10, dtype=int)
    orient_diag = regionprops(diag)[0].orientation
    assert_almost_equal(orient_diag, math.pi / 4)
    orient_diag = regionprops(diag, spacing=(1, 2))[0].orientation
    assert_almost_equal(orient_diag, np.arccos(0.5 / np.sqrt(1 + 0.5 ** 2)))
    orient_diag = regionprops(np.flipud(diag))[0].orientation
    assert_almost_equal(orient_diag, -math.pi / 4)
    orient_diag = regionprops(np.flipud(diag), spacing=(1, 2))[0].orientation
    assert_almost_equal(orient_diag, -np.arccos(0.5 / np.sqrt(1 + 0.5 ** 2)))
    orient_diag = regionprops(np.fliplr(diag))[0].orientation
    assert_almost_equal(orient_diag, -math.pi / 4)
    orient_diag = regionprops(np.fliplr(diag), spacing=(1, 2))[0].orientation
    assert_almost_equal(orient_diag, -np.arccos(0.5 / np.sqrt(1 + 0.5 ** 2)))
    orient_diag = regionprops(np.fliplr(np.flipud(diag)))[0].orientation
    assert_almost_equal(orient_diag, math.pi / 4)
    orient_diag = regionprops(np.fliplr(np.flipud(diag)), spacing=(1, 2))[0].orientation
    assert_almost_equal(orient_diag, np.arccos(0.5 / np.sqrt(1 + 0.5 ** 2)))