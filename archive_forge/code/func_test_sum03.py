import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_sum03():
    for type in types:
        input = np.ones([], type)
        output = ndimage.sum(input)
        assert_almost_equal(output, 1.0)