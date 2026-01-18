import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
@pytest.mark.parametrize('val1, val2', [(9007199254740993, 1), (9007199254740999, -2)])
def test_timedelta_floor_div_precision(self, val1, val2):
    op1 = np.timedelta64(val1)
    op2 = np.timedelta64(val2)
    actual = op1 // op2
    expected = val1 // val2
    assert_equal(actual, expected)