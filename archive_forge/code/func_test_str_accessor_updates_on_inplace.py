import pytest
from pandas import Series
import pandas._testing as tm
def test_str_accessor_updates_on_inplace(self):
    ser = Series(list('abc'))
    return_value = ser.drop([0], inplace=True)
    assert return_value is None
    assert len(ser.str.lower()) == 2