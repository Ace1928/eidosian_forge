from datetime import timedelta
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
from pandas import (
import pandas._testing as tm
def test_series_box_timestamp():
    rng = date_range('20090415', '20090519', freq='B')
    ser = Series(rng)
    assert isinstance(ser[0], Timestamp)
    assert isinstance(ser.at[1], Timestamp)
    assert isinstance(ser.iat[2], Timestamp)
    assert isinstance(ser.loc[3], Timestamp)
    assert isinstance(ser.iloc[4], Timestamp)
    ser = Series(rng, index=rng)
    msg = 'Series.__getitem__ treating keys as positions is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert isinstance(ser[0], Timestamp)
    assert isinstance(ser.at[rng[1]], Timestamp)
    assert isinstance(ser.iat[2], Timestamp)
    assert isinstance(ser.loc[rng[3]], Timestamp)
    assert isinstance(ser.iloc[4], Timestamp)