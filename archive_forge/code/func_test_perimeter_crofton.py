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
def test_perimeter_crofton():
    per = regionprops(SAMPLE)[0].perimeter_crofton
    target_per_crof = 61.0800637973
    assert_almost_equal(per, target_per_crof)
    per = regionprops(SAMPLE, spacing=(2, 2))[0].perimeter_crofton
    assert_almost_equal(per, 2 * target_per_crof)
    per = perimeter_crofton(SAMPLE.astype('double'), directions=2)
    assert_almost_equal(per, 64.4026493985)
    with testing.raises(NotImplementedError):
        per = regionprops(SAMPLE, spacing=(2, 1))[0].perimeter_crofton