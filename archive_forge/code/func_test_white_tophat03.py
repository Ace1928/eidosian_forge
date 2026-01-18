import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
def test_white_tophat03(self):
    array = numpy.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 0, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 1]], dtype=numpy.bool_)
    structure = numpy.ones((3, 3), dtype=numpy.bool_)
    expected = numpy.array([[0, 1, 1, 0, 0, 0, 0], [1, 0, 0, 1, 1, 1, 0], [1, 0, 0, 1, 1, 1, 0], [0, 1, 1, 0, 0, 0, 1], [0, 1, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 0, 1], [0, 0, 0, 1, 1, 1, 1]], dtype=numpy.bool_)
    output = ndimage.white_tophat(array, structure=structure)
    assert_array_equal(expected, output)