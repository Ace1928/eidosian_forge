from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
class TestIndexConstructorUnwrapping:

    @pytest.mark.parametrize('klass', [Index, DatetimeIndex])
    def test_constructor_from_series_dt64(self, klass):
        stamps = [Timestamp('20110101'), Timestamp('20120101'), Timestamp('20130101')]
        expected = DatetimeIndex(stamps)
        ser = Series(stamps)
        result = klass(ser)
        tm.assert_index_equal(result, expected)

    def test_constructor_no_pandas_array(self):
        ser = Series([1, 2, 3])
        result = Index(ser.array)
        expected = Index([1, 2, 3])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('array', [np.arange(5), np.array(['a', 'b', 'c']), date_range('2000-01-01', periods=3).values])
    def test_constructor_ndarray_like(self, array):

        class ArrayLike:

            def __init__(self, array) -> None:
                self.array = array

            def __array__(self, dtype=None) -> np.ndarray:
                return self.array
        expected = Index(array)
        result = Index(ArrayLike(array))
        tm.assert_index_equal(result, expected)