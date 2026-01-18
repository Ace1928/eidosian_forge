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
def test_perimeter():
    per = regionprops(SAMPLE)[0].perimeter
    target_per = 55.2487373415
    assert_almost_equal(per, target_per)
    per = regionprops(SAMPLE, spacing=(2, 2))[0].perimeter
    assert_almost_equal(per, 2 * target_per)
    per = perimeter(SAMPLE.astype('double'), neighborhood=8)
    assert_almost_equal(per, 46.8284271247)
    with testing.raises(NotImplementedError):
        per = regionprops(SAMPLE, spacing=(2, 1))[0].perimeter