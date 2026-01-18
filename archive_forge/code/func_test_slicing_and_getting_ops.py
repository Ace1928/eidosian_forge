import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_slicing_and_getting_ops(self):
    cats = Categorical(['a', 'c', 'b', 'c', 'c', 'c', 'c'], categories=['a', 'b', 'c'])
    idx = Index(['h', 'i', 'j', 'k', 'l', 'm', 'n'])
    values = [1, 2, 3, 4, 5, 6, 7]
    df = DataFrame({'cats': cats, 'values': values}, index=idx)
    cats2 = Categorical(['b', 'c'], categories=['a', 'b', 'c'])
    idx2 = Index(['j', 'k'])
    values2 = [3, 4]
    exp_df = DataFrame({'cats': cats2, 'values': values2}, index=idx2)
    exp_col = Series(cats, index=idx, name='cats')
    exp_row = Series(['b', 3], index=['cats', 'values'], dtype='object', name='j')
    exp_val = 'b'
    res_df = df.iloc[2:4, :]
    tm.assert_frame_equal(res_df, exp_df)
    assert isinstance(res_df['cats'].dtype, CategoricalDtype)
    res_row = df.iloc[2, :]
    tm.assert_series_equal(res_row, exp_row)
    assert isinstance(res_row['cats'], str)
    res_col = df.iloc[:, 0]
    tm.assert_series_equal(res_col, exp_col)
    assert isinstance(res_col.dtype, CategoricalDtype)
    res_val = df.iloc[2, 0]
    assert res_val == exp_val
    res_df = df.loc['j':'k', :]
    tm.assert_frame_equal(res_df, exp_df)
    assert isinstance(res_df['cats'].dtype, CategoricalDtype)
    res_row = df.loc['j', :]
    tm.assert_series_equal(res_row, exp_row)
    assert isinstance(res_row['cats'], str)
    res_col = df.loc[:, 'cats']
    tm.assert_series_equal(res_col, exp_col)
    assert isinstance(res_col.dtype, CategoricalDtype)
    res_val = df.loc['j', 'cats']
    assert res_val == exp_val
    res_val = df.loc['j', df.columns[0]]
    assert res_val == exp_val
    res_val = df.iat[2, 0]
    assert res_val == exp_val
    res_val = df.at['j', 'cats']
    assert res_val == exp_val
    exp_fancy = df.iloc[[2]]
    res_fancy = df[df['cats'] == 'b']
    tm.assert_frame_equal(res_fancy, exp_fancy)
    res_fancy = df[df['values'] == 3]
    tm.assert_frame_equal(res_fancy, exp_fancy)
    res_val = df.at['j', 'cats']
    assert res_val == exp_val
    res_row = df.iloc[2]
    tm.assert_series_equal(res_row, exp_row)
    assert isinstance(res_row['cats'], str)
    res_df = df.iloc[slice(2, 4)]
    tm.assert_frame_equal(res_df, exp_df)
    assert isinstance(res_df['cats'].dtype, CategoricalDtype)
    res_df = df.iloc[[2, 3]]
    tm.assert_frame_equal(res_df, exp_df)
    assert isinstance(res_df['cats'].dtype, CategoricalDtype)
    res_col = df.iloc[:, 0]
    tm.assert_series_equal(res_col, exp_col)
    assert isinstance(res_col.dtype, CategoricalDtype)
    res_df = df.iloc[:, slice(0, 2)]
    tm.assert_frame_equal(res_df, df)
    assert isinstance(res_df['cats'].dtype, CategoricalDtype)
    res_df = df.iloc[:, [0, 1]]
    tm.assert_frame_equal(res_df, df)
    assert isinstance(res_df['cats'].dtype, CategoricalDtype)