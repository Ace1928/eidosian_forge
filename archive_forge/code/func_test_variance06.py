import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_variance06():
    labels = [2, 2, 3, 3, 4]
    with np.errstate(all='ignore'):
        for type in types:
            input = np.array([1, 3, 8, 10, 8], type)
            output = ndimage.variance(input, labels, [2, 3, 4])
            assert_array_almost_equal(output, [1.0, 1.0, 0.0])