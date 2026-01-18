import colorsys
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import fetch, assert_stacklevel
from skimage._shared.utils import _supported_float_type, slice_at_axis
from skimage.color import (
from skimage.util import img_as_float, img_as_ubyte, img_as_float32
@pytest.mark.parametrize('func', [lab2rgb, lab2xyz])
def test_warning_stacklevel(self, func):
    regex = 'Conversion from CIE-LAB.* XYZ.*color space resulted in 1 negative Z values that have been clipped to zero'
    with pytest.warns(UserWarning, match=regex) as messages:
        func(lab=[[[0, 0, 300.0]]])
    assert_stacklevel(messages)
    assert len(messages) == 1
    assert messages[0].filename == __file__, 'warning points at wrong file'