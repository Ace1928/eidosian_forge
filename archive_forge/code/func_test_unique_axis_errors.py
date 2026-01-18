import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
def test_unique_axis_errors(self):
    assert_raises(TypeError, self._run_axis_tests, object)
    assert_raises(TypeError, self._run_axis_tests, [('a', int), ('b', object)])
    assert_raises(np.AxisError, unique, np.arange(10), axis=2)
    assert_raises(np.AxisError, unique, np.arange(10), axis=-2)