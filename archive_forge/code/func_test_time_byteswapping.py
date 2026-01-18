import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
@pytest.mark.parametrize('time_dtype', ['m8[D]', 'M8[Y]'])
def test_time_byteswapping(self, time_dtype):
    times = np.array(['2017', 'NaT'], dtype=time_dtype)
    times_swapped = times.astype(times.dtype.newbyteorder())
    assert_array_equal(times, times_swapped)
    unswapped = times_swapped.view(np.int64).newbyteorder()
    assert_array_equal(unswapped, times.view(np.int64))