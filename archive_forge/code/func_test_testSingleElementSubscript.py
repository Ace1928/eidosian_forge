from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_testSingleElementSubscript(self):
    a = array([1, 3, 2])
    b = array([1, 3, 2], mask=[1, 0, 1])
    assert_equal(a[0].shape, ())
    assert_equal(b[0].shape, ())
    assert_equal(b[1].shape, ())