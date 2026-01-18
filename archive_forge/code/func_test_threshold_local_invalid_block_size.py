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
@pytest.mark.parametrize('block_size', [(3,), (3, 3, 3)])
def test_threshold_local_invalid_block_size(self, block_size):
    with pytest.raises(ValueError):
        threshold_local(self.image, block_size, method='mean')