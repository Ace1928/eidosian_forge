import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_sum06():
    labels = np.array([], bool)
    for type in types:
        input = np.array([], type)
        output = ndimage.sum(input, labels=labels)
        assert_equal(output, 0.0)