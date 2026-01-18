import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
from skimage.color.delta_e import (
def test_single_color_cie76():
    lab1 = (0.5, 0.5, 0.5)
    lab2 = (0.4, 0.4, 0.4)
    deltaE_cie76(lab1, lab2)