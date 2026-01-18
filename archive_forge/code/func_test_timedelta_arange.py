import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_timedelta_arange(self):
    a = np.arange(3, 10, dtype='m8')
    assert_equal(a.dtype, np.dtype('m8'))
    assert_equal(a, np.timedelta64(0) + np.arange(3, 10))
    a = np.arange(np.timedelta64(3, 's'), 10, 2, dtype='m8')
    assert_equal(a.dtype, np.dtype('m8[s]'))
    assert_equal(a, np.timedelta64(0, 's') + np.arange(3, 10, 2))
    assert_raises(ValueError, np.arange, np.timedelta64(0), np.timedelta64(5), 0)
    assert_raises(TypeError, np.arange, np.timedelta64(0, 'D'), np.timedelta64(5, 'M'))
    assert_raises(TypeError, np.arange, np.timedelta64(0, 'Y'), np.timedelta64(5, 'D'))