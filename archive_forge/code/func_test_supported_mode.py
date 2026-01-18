import numpy as np
import pytest
from scipy import ndimage as ndi
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from skimage import color, data, transform
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import fetch, assert_stacklevel
from skimage.morphology import gray, footprints
from skimage.util import img_as_uint, img_as_ubyte
@pytest.mark.parametrize('func', gray_morphology_funcs)
@pytest.mark.parametrize('mode', gray._SUPPORTED_MODES)
def test_supported_mode(self, func, mode):
    img = np.ones((10, 10))
    func(img, mode=mode)