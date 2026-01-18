import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_timedelta_scalar_construction_units(self):
    assert_equal(np.datetime64('2010').dtype, np.dtype('M8[Y]'))
    assert_equal(np.datetime64('2010-03').dtype, np.dtype('M8[M]'))
    assert_equal(np.datetime64('2010-03-12').dtype, np.dtype('M8[D]'))
    assert_equal(np.datetime64('2010-03-12T17').dtype, np.dtype('M8[h]'))
    assert_equal(np.datetime64('2010-03-12T17:15').dtype, np.dtype('M8[m]'))
    assert_equal(np.datetime64('2010-03-12T17:15:08').dtype, np.dtype('M8[s]'))
    assert_equal(np.datetime64('2010-03-12T17:15:08.1').dtype, np.dtype('M8[ms]'))
    assert_equal(np.datetime64('2010-03-12T17:15:08.12').dtype, np.dtype('M8[ms]'))
    assert_equal(np.datetime64('2010-03-12T17:15:08.123').dtype, np.dtype('M8[ms]'))
    assert_equal(np.datetime64('2010-03-12T17:15:08.1234').dtype, np.dtype('M8[us]'))
    assert_equal(np.datetime64('2010-03-12T17:15:08.12345').dtype, np.dtype('M8[us]'))
    assert_equal(np.datetime64('2010-03-12T17:15:08.123456').dtype, np.dtype('M8[us]'))
    assert_equal(np.datetime64('1970-01-01T00:00:02.1234567').dtype, np.dtype('M8[ns]'))
    assert_equal(np.datetime64('1970-01-01T00:00:02.12345678').dtype, np.dtype('M8[ns]'))
    assert_equal(np.datetime64('1970-01-01T00:00:02.123456789').dtype, np.dtype('M8[ns]'))
    assert_equal(np.datetime64('1970-01-01T00:00:02.1234567890').dtype, np.dtype('M8[ps]'))
    assert_equal(np.datetime64('1970-01-01T00:00:02.12345678901').dtype, np.dtype('M8[ps]'))
    assert_equal(np.datetime64('1970-01-01T00:00:02.123456789012').dtype, np.dtype('M8[ps]'))
    assert_equal(np.datetime64('1970-01-01T00:00:02.1234567890123').dtype, np.dtype('M8[fs]'))
    assert_equal(np.datetime64('1970-01-01T00:00:02.12345678901234').dtype, np.dtype('M8[fs]'))
    assert_equal(np.datetime64('1970-01-01T00:00:02.123456789012345').dtype, np.dtype('M8[fs]'))
    assert_equal(np.datetime64('1970-01-01T00:00:02.1234567890123456').dtype, np.dtype('M8[as]'))
    assert_equal(np.datetime64('1970-01-01T00:00:02.12345678901234567').dtype, np.dtype('M8[as]'))
    assert_equal(np.datetime64('1970-01-01T00:00:02.123456789012345678').dtype, np.dtype('M8[as]'))
    assert_equal(np.datetime64(datetime.date(2010, 4, 16)).dtype, np.dtype('M8[D]'))
    assert_equal(np.datetime64(datetime.datetime(2010, 4, 16, 13, 45, 18)).dtype, np.dtype('M8[us]'))
    assert_equal(np.datetime64('today').dtype, np.dtype('M8[D]'))
    assert_equal(np.datetime64('now').dtype, np.dtype('M8[s]'))