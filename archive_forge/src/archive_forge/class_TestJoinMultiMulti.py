import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import merge
class TestJoinMultiMulti:

    def test_join_multi_multi(self, left_multi, right_multi, join_type, on_cols_multi):
        left_names = left_multi.index.names
        right_names = right_multi.index.names
        if join_type == 'right':
            level_order = right_names + left_names.difference(right_names)
        else:
            level_order = left_names + right_names.difference(left_names)
        expected = merge(left_multi.reset_index(), right_multi.reset_index(), how=join_type, on=on_cols_multi).set_index(level_order).sort_index()
        result = left_multi.join(right_multi, how=join_type).sort_index()
        tm.assert_frame_equal(result, expected)

    def test_join_multi_empty_frames(self, left_multi, right_multi, join_type, on_cols_multi):
        left_multi = left_multi.drop(columns=left_multi.columns)
        right_multi = right_multi.drop(columns=right_multi.columns)
        left_names = left_multi.index.names
        right_names = right_multi.index.names
        if join_type == 'right':
            level_order = right_names + left_names.difference(right_names)
        else:
            level_order = left_names + right_names.difference(left_names)
        expected = merge(left_multi.reset_index(), right_multi.reset_index(), how=join_type, on=on_cols_multi).set_index(level_order).sort_index()
        result = left_multi.join(right_multi, how=join_type).sort_index()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('box', [None, np.asarray, Series, Index])
    def test_merge_datetime_index(self, box):
        df = DataFrame([1, 2, 3], ['2016-01-01', '2017-01-01', '2018-01-01'], columns=['a'])
        df.index = pd.to_datetime(df.index)
        on_vector = df.index.year
        if box is not None:
            on_vector = box(on_vector)
        exp_years = np.array([2016, 2017, 2018], dtype=np.int32)
        expected = DataFrame({'a': [1, 2, 3], 'key_1': exp_years})
        result = df.merge(df, on=['a', on_vector], how='inner')
        tm.assert_frame_equal(result, expected)
        expected = DataFrame({'key_0': exp_years, 'a_x': [1, 2, 3], 'a_y': [1, 2, 3]})
        result = df.merge(df, on=[df.index.year], how='inner')
        tm.assert_frame_equal(result, expected)

    def test_single_common_level(self):
        index_left = MultiIndex.from_tuples([('K0', 'X0'), ('K0', 'X1'), ('K1', 'X2')], names=['key', 'X'])
        left = DataFrame({'A': ['A0', 'A1', 'A2'], 'B': ['B0', 'B1', 'B2']}, index=index_left)
        index_right = MultiIndex.from_tuples([('K0', 'Y0'), ('K1', 'Y1'), ('K2', 'Y2'), ('K2', 'Y3')], names=['key', 'Y'])
        right = DataFrame({'C': ['C0', 'C1', 'C2', 'C3'], 'D': ['D0', 'D1', 'D2', 'D3']}, index=index_right)
        result = left.join(right)
        expected = merge(left.reset_index(), right.reset_index(), on=['key'], how='inner').set_index(['key', 'X', 'Y'])
        tm.assert_frame_equal(result, expected)

    def test_join_multi_wrong_order(self):
        midx1 = MultiIndex.from_product([[1, 2], [3, 4]], names=['a', 'b'])
        midx3 = MultiIndex.from_tuples([(4, 1), (3, 2), (3, 1)], names=['b', 'a'])
        left = DataFrame(index=midx1, data={'x': [10, 20, 30, 40]})
        right = DataFrame(index=midx3, data={'y': ['foo', 'bar', 'fing']})
        result = left.join(right)
        expected = DataFrame(index=midx1, data={'x': [10, 20, 30, 40], 'y': ['fing', 'foo', 'bar', np.nan]})
        tm.assert_frame_equal(result, expected)