from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestLocILocDataFrameCategorical:

    @pytest.fixture
    def orig(self):
        cats = Categorical(['a', 'a', 'a', 'a', 'a', 'a', 'a'], categories=['a', 'b'])
        idx = Index(['h', 'i', 'j', 'k', 'l', 'm', 'n'])
        values = [1, 1, 1, 1, 1, 1, 1]
        orig = DataFrame({'cats': cats, 'values': values}, index=idx)
        return orig

    @pytest.fixture
    def exp_single_row(self):
        cats1 = Categorical(['a', 'a', 'b', 'a', 'a', 'a', 'a'], categories=['a', 'b'])
        idx1 = Index(['h', 'i', 'j', 'k', 'l', 'm', 'n'])
        values1 = [1, 1, 2, 1, 1, 1, 1]
        exp_single_row = DataFrame({'cats': cats1, 'values': values1}, index=idx1)
        return exp_single_row

    @pytest.fixture
    def exp_multi_row(self):
        cats2 = Categorical(['a', 'a', 'b', 'b', 'a', 'a', 'a'], categories=['a', 'b'])
        idx2 = Index(['h', 'i', 'j', 'k', 'l', 'm', 'n'])
        values2 = [1, 1, 2, 2, 1, 1, 1]
        exp_multi_row = DataFrame({'cats': cats2, 'values': values2}, index=idx2)
        return exp_multi_row

    @pytest.fixture
    def exp_parts_cats_col(self):
        cats3 = Categorical(['a', 'a', 'b', 'b', 'a', 'a', 'a'], categories=['a', 'b'])
        idx3 = Index(['h', 'i', 'j', 'k', 'l', 'm', 'n'])
        values3 = [1, 1, 1, 1, 1, 1, 1]
        exp_parts_cats_col = DataFrame({'cats': cats3, 'values': values3}, index=idx3)
        return exp_parts_cats_col

    @pytest.fixture
    def exp_single_cats_value(self):
        cats4 = Categorical(['a', 'a', 'b', 'a', 'a', 'a', 'a'], categories=['a', 'b'])
        idx4 = Index(['h', 'i', 'j', 'k', 'l', 'm', 'n'])
        values4 = [1, 1, 1, 1, 1, 1, 1]
        exp_single_cats_value = DataFrame({'cats': cats4, 'values': values4}, index=idx4)
        return exp_single_cats_value

    @pytest.mark.parametrize('indexer', [tm.loc, tm.iloc])
    def test_loc_iloc_setitem_list_of_lists(self, orig, exp_multi_row, indexer):
        df = orig.copy()
        key = slice(2, 4)
        if indexer is tm.loc:
            key = slice('j', 'k')
        indexer(df)[key, :] = [['b', 2], ['b', 2]]
        tm.assert_frame_equal(df, exp_multi_row)
        df = orig.copy()
        with pytest.raises(TypeError, match=msg1):
            indexer(df)[key, :] = [['c', 2], ['c', 2]]

    @pytest.mark.parametrize('indexer', [tm.loc, tm.iloc, tm.at, tm.iat])
    def test_loc_iloc_at_iat_setitem_single_value_in_categories(self, orig, exp_single_cats_value, indexer):
        df = orig.copy()
        key = (2, 0)
        if indexer in [tm.loc, tm.at]:
            key = (df.index[2], df.columns[0])
        indexer(df)[key] = 'b'
        tm.assert_frame_equal(df, exp_single_cats_value)
        with pytest.raises(TypeError, match=msg1):
            indexer(df)[key] = 'c'

    @pytest.mark.parametrize('indexer', [tm.loc, tm.iloc])
    def test_loc_iloc_setitem_mask_single_value_in_categories(self, orig, exp_single_cats_value, indexer):
        df = orig.copy()
        mask = df.index == 'j'
        key = 0
        if indexer is tm.loc:
            key = df.columns[key]
        indexer(df)[mask, key] = 'b'
        tm.assert_frame_equal(df, exp_single_cats_value)

    @pytest.mark.parametrize('indexer', [tm.loc, tm.iloc])
    def test_loc_iloc_setitem_full_row_non_categorical_rhs(self, orig, exp_single_row, indexer):
        df = orig.copy()
        key = 2
        if indexer is tm.loc:
            key = df.index[2]
        indexer(df)[key, :] = ['b', 2]
        tm.assert_frame_equal(df, exp_single_row)
        with pytest.raises(TypeError, match=msg1):
            indexer(df)[key, :] = ['c', 2]

    @pytest.mark.parametrize('indexer', [tm.loc, tm.iloc])
    def test_loc_iloc_setitem_partial_col_categorical_rhs(self, orig, exp_parts_cats_col, indexer):
        df = orig.copy()
        key = (slice(2, 4), 0)
        if indexer is tm.loc:
            key = (slice('j', 'k'), df.columns[0])
        compat = Categorical(['b', 'b'], categories=['a', 'b'])
        indexer(df)[key] = compat
        tm.assert_frame_equal(df, exp_parts_cats_col)
        semi_compat = Categorical(list('bb'), categories=list('abc'))
        with pytest.raises(TypeError, match=msg2):
            indexer(df)[key] = semi_compat
        incompat = Categorical(list('cc'), categories=list('abc'))
        with pytest.raises(TypeError, match=msg2):
            indexer(df)[key] = incompat

    @pytest.mark.parametrize('indexer', [tm.loc, tm.iloc])
    def test_loc_iloc_setitem_non_categorical_rhs(self, orig, exp_parts_cats_col, indexer):
        df = orig.copy()
        key = (slice(2, 4), 0)
        if indexer is tm.loc:
            key = (slice('j', 'k'), df.columns[0])
        indexer(df)[key] = ['b', 'b']
        tm.assert_frame_equal(df, exp_parts_cats_col)
        with pytest.raises(TypeError, match=msg1):
            indexer(df)[key] = ['c', 'c']

    @pytest.mark.parametrize('indexer', [tm.getitem, tm.loc, tm.iloc])
    def test_getitem_preserve_object_index_with_dates(self, indexer):
        idx = date_range('2012', periods=3).astype(object)
        df = DataFrame({0: [1, 2, 3]}, index=idx)
        assert df.index.dtype == object
        if indexer is tm.getitem:
            ser = indexer(df)[0]
        else:
            ser = indexer(df)[:, 0]
        assert ser.index.dtype == object

    def test_loc_on_multiindex_one_level(self):
        df = DataFrame(data=[[0], [1]], index=MultiIndex.from_tuples([('a',), ('b',)], names=['first']))
        expected = DataFrame(data=[[0]], index=MultiIndex.from_tuples([('a',)], names=['first']))
        result = df.loc['a']
        tm.assert_frame_equal(result, expected)