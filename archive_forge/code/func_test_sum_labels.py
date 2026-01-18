import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_sum_labels():
    labels = np.array([[1, 2], [2, 4]], np.int8)
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output_sum = ndimage.sum(input, labels=labels, index=[4, 8, 2])
        output_labels = ndimage.sum_labels(input, labels=labels, index=[4, 8, 2])
        assert (output_sum == output_labels).all()
        assert_array_almost_equal(output_labels, [4.0, 0.0, 5.0])