import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_pydatetime_creation(self):
    a = np.array(['1960-03-12', datetime.date(1960, 3, 12)], dtype='M8[D]')
    assert_equal(a[0], a[1])
    a = np.array(['1999-12-31', datetime.date(1999, 12, 31)], dtype='M8[D]')
    assert_equal(a[0], a[1])
    a = np.array(['2000-01-01', datetime.date(2000, 1, 1)], dtype='M8[D]')
    assert_equal(a[0], a[1])
    a = np.array(['today', datetime.date.today()], dtype='M8[D]')
    assert_equal(a[0], a[1])
    assert_equal(np.array(datetime.date(1960, 3, 12), dtype='M8[s]'), np.array(np.datetime64('1960-03-12T00:00:00')))