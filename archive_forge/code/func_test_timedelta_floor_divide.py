import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
@pytest.mark.parametrize('op1, op2, exp', [(np.timedelta64(7, 's'), np.timedelta64(4, 's'), 1), (np.timedelta64(7, 's'), np.timedelta64(-4, 's'), -2), (np.timedelta64(8, 's'), np.timedelta64(-4, 's'), -2), (np.timedelta64(1, 'm'), np.timedelta64(31, 's'), 1), (np.timedelta64(1890), np.timedelta64(31), 60), (np.timedelta64(2, 'Y'), np.timedelta64('13', 'M'), 1), (np.array([1, 2, 3], dtype='m8'), np.array([2], dtype='m8'), np.array([0, 1, 1], dtype=np.int64))])
def test_timedelta_floor_divide(self, op1, op2, exp):
    assert_equal(op1 // op2, exp)