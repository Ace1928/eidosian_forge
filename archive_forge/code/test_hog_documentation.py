import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import color, data, draw, feature, img_as_float
from skimage._shared import filters
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
Test that an exception is thrown whenever the input image is
    too small for the given parameters.
    