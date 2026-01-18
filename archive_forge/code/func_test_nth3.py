import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_nth3():
    df = DataFrame(np.random.default_rng(2).integers(1, 10, (100, 2)), dtype='int64')
    ser = df[1]
    gb = df[0]
    expected = ser.groupby(gb).first()
    expected2 = ser.groupby(gb).apply(lambda x: x.iloc[0])
    tm.assert_series_equal(expected2, expected, check_names=False)
    assert expected.name == 1
    assert expected2.name == 1
    v = ser[gb == 1].iloc[0]
    assert expected.iloc[0] == v
    assert expected2.iloc[0] == v
    with pytest.raises(ValueError, match='For a DataFrame'):
        ser.groupby(gb, sort=False).nth(0, dropna=True)