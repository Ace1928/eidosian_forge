import re
import numpy as np
import pytest
from pandas.core.dtypes import generic as gt
import pandas as pd
import pandas._testing as tm
def test_setattr_warnings():
    d = {'one': pd.Series([1.0, 2.0, 3.0], index=['a', 'b', 'c']), 'two': pd.Series([1.0, 2.0, 3.0, 4.0], index=['a', 'b', 'c', 'd'])}
    df = pd.DataFrame(d)
    with tm.assert_produces_warning(None):
        df['three'] = df.two + 1
        assert df.three.sum() > df.two.sum()
    with tm.assert_produces_warning(None):
        df.one += 1
        assert df.one.iloc[0] == 2
    with tm.assert_produces_warning(None):
        df.two.not_an_index = [1, 2]
    with tm.assert_produces_warning(UserWarning):
        df.four = df.two + 2
        assert df.four.sum() > df.two.sum()