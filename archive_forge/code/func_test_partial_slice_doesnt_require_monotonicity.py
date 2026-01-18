import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_partial_slice_doesnt_require_monotonicity(self):
    dti = date_range('2014-01-01', periods=30, freq='30D')
    pi = dti.to_period('D')
    ser_montonic = Series(np.arange(30), index=pi)
    shuffler = list(range(0, 30, 2)) + list(range(1, 31, 2))
    ser = ser_montonic.iloc[shuffler]
    nidx = ser.index
    indexer_2014 = np.array([0, 1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19, 20], dtype=np.intp)
    assert (nidx[indexer_2014].year == 2014).all()
    assert not (nidx[~indexer_2014].year == 2014).any()
    result = nidx.get_loc('2014')
    tm.assert_numpy_array_equal(result, indexer_2014)
    expected = ser.iloc[indexer_2014]
    result = ser.loc['2014']
    tm.assert_series_equal(result, expected)
    result = ser['2014']
    tm.assert_series_equal(result, expected)
    indexer_may2015 = np.array([23], dtype=np.intp)
    assert nidx[23].year == 2015 and nidx[23].month == 5
    result = nidx.get_loc('May 2015')
    tm.assert_numpy_array_equal(result, indexer_may2015)
    expected = ser.iloc[indexer_may2015]
    result = ser.loc['May 2015']
    tm.assert_series_equal(result, expected)
    result = ser['May 2015']
    tm.assert_series_equal(result, expected)