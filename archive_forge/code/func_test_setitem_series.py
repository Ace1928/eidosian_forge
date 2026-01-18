import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_setitem_series(self, data, full_indexer):
    ser = pd.Series(data, name='data')
    result = pd.Series(index=ser.index, dtype=object, name='data')
    key = full_indexer(ser)
    result.loc[key] = ser
    expected = pd.Series(data.astype(object), index=ser.index, name='data', dtype=object)
    tm.assert_series_equal(result, expected)