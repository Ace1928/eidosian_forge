import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_timedelta_0_dim_object_array_conversion(self):
    test = np.array(datetime.timedelta(seconds=20))
    actual = test.astype(np.timedelta64)
    expected = np.array(datetime.timedelta(seconds=20), np.timedelta64)
    assert_equal(actual, expected)