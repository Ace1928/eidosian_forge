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
@pytest.mark.parametrize('ndim', [2, 3])
def test_threshold_local_mean(self, ndim):
    ref = np.array([[False, False, False, False, True], [False, False, True, False, True], [False, False, True, True, False], [False, True, True, False, False], [True, True, False, False, False]])
    if ndim == 2:
        image = self.image
        block_sizes = [3, (3,) * image.ndim]
    else:
        image = np.stack((self.image,) * 5, axis=-1)
        ref = np.stack((ref,) * 5, axis=-1)
        block_sizes = [3, (3,) * image.ndim, (3,) * (image.ndim - 1) + (1,)]
    for block_size in block_sizes:
        out = threshold_local(image, block_size, method='mean', mode='reflect')
        assert_equal(ref, image > out)