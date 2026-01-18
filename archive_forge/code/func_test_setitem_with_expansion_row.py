import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_setitem_with_expansion_row(self, data, na_value):
    df = pd.DataFrame({'data': data[:1]})
    df.loc[1, 'data'] = data[1]
    expected = pd.DataFrame({'data': data[:2]})
    tm.assert_frame_equal(df, expected)
    df.loc[2, 'data'] = na_value
    expected = pd.DataFrame({'data': pd.Series([data[0], data[1], na_value], dtype=data.dtype)})
    tm.assert_frame_equal(df, expected)