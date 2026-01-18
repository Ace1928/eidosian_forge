import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_standard_deviation03():
    for type in types:
        input = np.array([1, 3], type)
        output = ndimage.standard_deviation(input)
        assert_almost_equal(output, np.sqrt(1.0))