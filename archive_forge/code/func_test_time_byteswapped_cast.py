import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
@pytest.mark.parametrize(['time1', 'time2'], [('M8[s]', 'M8[D]'), ('m8[s]', 'm8[ns]')])
def test_time_byteswapped_cast(self, time1, time2):
    dtype1 = np.dtype(time1)
    dtype2 = np.dtype(time2)
    times = np.array(['2017', 'NaT'], dtype=dtype1)
    expected = times.astype(dtype2)
    res = times.astype(dtype1.newbyteorder()).astype(dtype2)
    assert_array_equal(res, expected)
    res = times.astype(dtype2.newbyteorder())
    assert_array_equal(res, expected)
    res = times.astype(dtype1.newbyteorder()).astype(dtype2.newbyteorder())
    assert_array_equal(res, expected)