import numpy as np
import pytest
from scipy import ndimage as ndi
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from skimage import color, data, transform
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import fetch, assert_stacklevel
from skimage.morphology import gray, footprints
from skimage.util import img_as_uint, img_as_ubyte
@pytest.mark.parametrize('function', ['erosion', 'dilation', 'closing', 'opening', 'white_tophat', 'black_tophat'])
@pytest.mark.parametrize('nrows', (3, 11))
@pytest.mark.parametrize('ncols', (3, 11))
@pytest.mark.parametrize('decomposition', ['separable', 'sequence'])
def test_rectangle_decomposition(cam_image, function, nrows, ncols, decomposition):
    """Validate footprint decomposition for various shapes.

    comparison is made to the case without decomposition.
    """
    footprint_ndarray = footprints.rectangle(nrows, ncols, decomposition=None)
    footprint = footprints.rectangle(nrows, ncols, decomposition=decomposition)
    func = getattr(gray, function)
    expected = func(cam_image, footprint=footprint_ndarray)
    out = func(cam_image, footprint=footprint)
    assert_array_equal(expected, out)