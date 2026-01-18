import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
@pytest.mark.parametrize('size', [3, 21, 217, 1000])
def test_timedelta_nat_argsort_stability(self, size):
    expected = np.arange(size)
    arr = np.tile(np.timedelta64('NaT'), size)
    assert_equal(np.argsort(arr, kind='mergesort'), expected)