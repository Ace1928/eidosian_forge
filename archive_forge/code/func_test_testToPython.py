from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_testToPython(self):
    assert_equal(1, int(array(1)))
    assert_equal(1.0, float(array(1)))
    assert_equal(1, int(array([[[1]]])))
    assert_equal(1.0, float(array([[1]])))
    assert_raises(TypeError, float, array([1, 1]))
    assert_raises(ValueError, bool, array([0, 1]))
    assert_raises(ValueError, bool, array([0, 0], mask=[0, 1]))