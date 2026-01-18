import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_timedelta_arange_no_dtype(self):
    d = np.array(5, dtype='m8[D]')
    assert_equal(np.arange(d, d + 1), d)
    assert_equal(np.arange(d), np.arange(0, d))