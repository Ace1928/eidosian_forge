import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
from skimage.color.delta_e import (
Test for correctness of color distance functions