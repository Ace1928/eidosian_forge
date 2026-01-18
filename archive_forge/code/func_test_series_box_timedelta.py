from datetime import timedelta
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
from pandas import (
import pandas._testing as tm
def test_series_box_timedelta():
    rng = timedelta_range('1 day 1 s', periods=5, freq='h')
    ser = Series(rng)
    assert isinstance(ser[0], Timedelta)
    assert isinstance(ser.at[1], Timedelta)
    assert isinstance(ser.iat[2], Timedelta)
    assert isinstance(ser.loc[3], Timedelta)
    assert isinstance(ser.iloc[4], Timedelta)