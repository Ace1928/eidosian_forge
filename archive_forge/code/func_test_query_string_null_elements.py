import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
@pytest.mark.parametrize('in_list', [[None, 'asdf', 'ghjk'], ['asdf', None, 'ghjk'], ['asdf', 'ghjk', None], [None, None, 'asdf'], ['asdf', None, None], [None, None, None]])
def test_query_string_null_elements(self, in_list):
    parser = 'pandas'
    engine = 'python'
    expected = {i: value for i, value in enumerate(in_list) if value == 'asdf'}
    df_expected = DataFrame({'a': expected}, dtype='string')
    df_expected.index = df_expected.index.astype('int64')
    df = DataFrame({'a': in_list}, dtype='string')
    res1 = df.query("a == 'asdf'", parser=parser, engine=engine)
    res2 = df[df['a'] == 'asdf']
    res3 = df.query("a <= 'asdf'", parser=parser, engine=engine)
    tm.assert_frame_equal(res1, df_expected)
    tm.assert_frame_equal(res1, res2)
    tm.assert_frame_equal(res1, res3)
    tm.assert_frame_equal(res2, res3)