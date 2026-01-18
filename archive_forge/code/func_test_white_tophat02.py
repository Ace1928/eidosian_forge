import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
def test_white_tophat02(self):
    array = numpy.array([[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]])
    footprint = [[1, 0, 1], [1, 1, 0]]
    structure = [[0, 0, 0], [0, 0, 0]]
    tmp = ndimage.grey_opening(array, footprint=footprint, structure=structure)
    expected = array - tmp
    output = ndimage.white_tophat(array, footprint=footprint, structure=structure)
    assert_array_almost_equal(expected, output)