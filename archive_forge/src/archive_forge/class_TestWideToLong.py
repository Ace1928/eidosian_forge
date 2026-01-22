import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestWideToLong:

    def test_simple(self):
        x = np.random.default_rng(2).standard_normal(3)
        df = DataFrame({'A1970': {0: 'a', 1: 'b', 2: 'c'}, 'A1980': {0: 'd', 1: 'e', 2: 'f'}, 'B1970': {0: 2.5, 1: 1.2, 2: 0.7}, 'B1980': {0: 3.2, 1: 1.3, 2: 0.1}, 'X': dict(zip(range(3), x))})
        df['id'] = df.index
        exp_data = {'X': x.tolist() + x.tolist(), 'A': ['a', 'b', 'c', 'd', 'e', 'f'], 'B': [2.5, 1.2, 0.7, 3.2, 1.3, 0.1], 'year': [1970, 1970, 1970, 1980, 1980, 1980], 'id': [0, 1, 2, 0, 1, 2]}
        expected = DataFrame(exp_data)
        expected = expected.set_index(['id', 'year'])[['X', 'A', 'B']]
        result = wide_to_long(df, ['A', 'B'], i='id', j='year')
        tm.assert_frame_equal(result, expected)

    def test_stubs(self):
        df = DataFrame([[0, 1, 2, 3, 8], [4, 5, 6, 7, 9]])
        df.columns = ['id', 'inc1', 'inc2', 'edu1', 'edu2']
        stubs = ['inc', 'edu']
        wide_to_long(df, stubs, i='id', j='age')
        assert stubs == ['inc', 'edu']

    def test_separating_character(self):
        x = np.random.default_rng(2).standard_normal(3)
        df = DataFrame({'A.1970': {0: 'a', 1: 'b', 2: 'c'}, 'A.1980': {0: 'd', 1: 'e', 2: 'f'}, 'B.1970': {0: 2.5, 1: 1.2, 2: 0.7}, 'B.1980': {0: 3.2, 1: 1.3, 2: 0.1}, 'X': dict(zip(range(3), x))})
        df['id'] = df.index
        exp_data = {'X': x.tolist() + x.tolist(), 'A': ['a', 'b', 'c', 'd', 'e', 'f'], 'B': [2.5, 1.2, 0.7, 3.2, 1.3, 0.1], 'year': [1970, 1970, 1970, 1980, 1980, 1980], 'id': [0, 1, 2, 0, 1, 2]}
        expected = DataFrame(exp_data)
        expected = expected.set_index(['id', 'year'])[['X', 'A', 'B']]
        result = wide_to_long(df, ['A', 'B'], i='id', j='year', sep='.')
        tm.assert_frame_equal(result, expected)

    def test_escapable_characters(self):
        x = np.random.default_rng(2).standard_normal(3)
        df = DataFrame({'A(quarterly)1970': {0: 'a', 1: 'b', 2: 'c'}, 'A(quarterly)1980': {0: 'd', 1: 'e', 2: 'f'}, 'B(quarterly)1970': {0: 2.5, 1: 1.2, 2: 0.7}, 'B(quarterly)1980': {0: 3.2, 1: 1.3, 2: 0.1}, 'X': dict(zip(range(3), x))})
        df['id'] = df.index
        exp_data = {'X': x.tolist() + x.tolist(), 'A(quarterly)': ['a', 'b', 'c', 'd', 'e', 'f'], 'B(quarterly)': [2.5, 1.2, 0.7, 3.2, 1.3, 0.1], 'year': [1970, 1970, 1970, 1980, 1980, 1980], 'id': [0, 1, 2, 0, 1, 2]}
        expected = DataFrame(exp_data)
        expected = expected.set_index(['id', 'year'])[['X', 'A(quarterly)', 'B(quarterly)']]
        result = wide_to_long(df, ['A(quarterly)', 'B(quarterly)'], i='id', j='year')
        tm.assert_frame_equal(result, expected)

    def test_unbalanced(self):
        df = DataFrame({'A2010': [1.0, 2.0], 'A2011': [3.0, 4.0], 'B2010': [5.0, 6.0], 'X': ['X1', 'X2']})
        df['id'] = df.index
        exp_data = {'X': ['X1', 'X2', 'X1', 'X2'], 'A': [1.0, 2.0, 3.0, 4.0], 'B': [5.0, 6.0, np.nan, np.nan], 'id': [0, 1, 0, 1], 'year': [2010, 2010, 2011, 2011]}
        expected = DataFrame(exp_data)
        expected = expected.set_index(['id', 'year'])[['X', 'A', 'B']]
        result = wide_to_long(df, ['A', 'B'], i='id', j='year')
        tm.assert_frame_equal(result, expected)

    def test_character_overlap(self):
        df = DataFrame({'A11': ['a11', 'a22', 'a33'], 'A12': ['a21', 'a22', 'a23'], 'B11': ['b11', 'b12', 'b13'], 'B12': ['b21', 'b22', 'b23'], 'BB11': [1, 2, 3], 'BB12': [4, 5, 6], 'BBBX': [91, 92, 93], 'BBBZ': [91, 92, 93]})
        df['id'] = df.index
        expected = DataFrame({'BBBX': [91, 92, 93, 91, 92, 93], 'BBBZ': [91, 92, 93, 91, 92, 93], 'A': ['a11', 'a22', 'a33', 'a21', 'a22', 'a23'], 'B': ['b11', 'b12', 'b13', 'b21', 'b22', 'b23'], 'BB': [1, 2, 3, 4, 5, 6], 'id': [0, 1, 2, 0, 1, 2], 'year': [11, 11, 11, 12, 12, 12]})
        expected = expected.set_index(['id', 'year'])[['BBBX', 'BBBZ', 'A', 'B', 'BB']]
        result = wide_to_long(df, ['A', 'B', 'BB'], i='id', j='year')
        tm.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))

    def test_invalid_separator(self):
        sep = 'nope!'
        df = DataFrame({'A2010': [1.0, 2.0], 'A2011': [3.0, 4.0], 'B2010': [5.0, 6.0], 'X': ['X1', 'X2']})
        df['id'] = df.index
        exp_data = {'X': '', 'A2010': [], 'A2011': [], 'B2010': [], 'id': [], 'year': [], 'A': [], 'B': []}
        expected = DataFrame(exp_data).astype({'year': np.int64})
        expected = expected.set_index(['id', 'year'])[['X', 'A2010', 'A2011', 'B2010', 'A', 'B']]
        expected.index = expected.index.set_levels([0, 1], level=0)
        result = wide_to_long(df, ['A', 'B'], i='id', j='year', sep=sep)
        tm.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))

    def test_num_string_disambiguation(self):
        df = DataFrame({'A11': ['a11', 'a22', 'a33'], 'A12': ['a21', 'a22', 'a23'], 'B11': ['b11', 'b12', 'b13'], 'B12': ['b21', 'b22', 'b23'], 'BB11': [1, 2, 3], 'BB12': [4, 5, 6], 'Arating': [91, 92, 93], 'Arating_old': [91, 92, 93]})
        df['id'] = df.index
        expected = DataFrame({'Arating': [91, 92, 93, 91, 92, 93], 'Arating_old': [91, 92, 93, 91, 92, 93], 'A': ['a11', 'a22', 'a33', 'a21', 'a22', 'a23'], 'B': ['b11', 'b12', 'b13', 'b21', 'b22', 'b23'], 'BB': [1, 2, 3, 4, 5, 6], 'id': [0, 1, 2, 0, 1, 2], 'year': [11, 11, 11, 12, 12, 12]})
        expected = expected.set_index(['id', 'year'])[['Arating', 'Arating_old', 'A', 'B', 'BB']]
        result = wide_to_long(df, ['A', 'B', 'BB'], i='id', j='year')
        tm.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))

    def test_invalid_suffixtype(self):
        df = DataFrame({'Aone': [1.0, 2.0], 'Atwo': [3.0, 4.0], 'Bone': [5.0, 6.0], 'X': ['X1', 'X2']})
        df['id'] = df.index
        exp_data = {'X': '', 'Aone': [], 'Atwo': [], 'Bone': [], 'id': [], 'year': [], 'A': [], 'B': []}
        expected = DataFrame(exp_data).astype({'year': np.int64})
        expected = expected.set_index(['id', 'year'])
        expected.index = expected.index.set_levels([0, 1], level=0)
        result = wide_to_long(df, ['A', 'B'], i='id', j='year')
        tm.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))

    def test_multiple_id_columns(self):
        df = DataFrame({'famid': [1, 1, 1, 2, 2, 2, 3, 3, 3], 'birth': [1, 2, 3, 1, 2, 3, 1, 2, 3], 'ht1': [2.8, 2.9, 2.2, 2, 1.8, 1.9, 2.2, 2.3, 2.1], 'ht2': [3.4, 3.8, 2.9, 3.2, 2.8, 2.4, 3.3, 3.4, 2.9]})
        expected = DataFrame({'ht': [2.8, 3.4, 2.9, 3.8, 2.2, 2.9, 2.0, 3.2, 1.8, 2.8, 1.9, 2.4, 2.2, 3.3, 2.3, 3.4, 2.1, 2.9], 'famid': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3], 'birth': [1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3], 'age': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]})
        expected = expected.set_index(['famid', 'birth', 'age'])[['ht']]
        result = wide_to_long(df, 'ht', i=['famid', 'birth'], j='age')
        tm.assert_frame_equal(result, expected)

    def test_non_unique_idvars(self):
        df = DataFrame({'A_A1': [1, 2, 3, 4, 5], 'B_B1': [1, 2, 3, 4, 5], 'x': [1, 1, 1, 1, 1]})
        msg = 'the id variables need to uniquely identify each row'
        with pytest.raises(ValueError, match=msg):
            wide_to_long(df, ['A_A', 'B_B'], i='x', j='colname')

    def test_cast_j_int(self):
        df = DataFrame({'actor_1': ['CCH Pounder', 'Johnny Depp', 'Christoph Waltz'], 'actor_2': ['Joel David Moore', 'Orlando Bloom', 'Rory Kinnear'], 'actor_fb_likes_1': [1000.0, 40000.0, 11000.0], 'actor_fb_likes_2': [936.0, 5000.0, 393.0], 'title': ['Avatar', 'Pirates of the Caribbean', 'Spectre']})
        expected = DataFrame({'actor': ['CCH Pounder', 'Johnny Depp', 'Christoph Waltz', 'Joel David Moore', 'Orlando Bloom', 'Rory Kinnear'], 'actor_fb_likes': [1000.0, 40000.0, 11000.0, 936.0, 5000.0, 393.0], 'num': [1, 1, 1, 2, 2, 2], 'title': ['Avatar', 'Pirates of the Caribbean', 'Spectre', 'Avatar', 'Pirates of the Caribbean', 'Spectre']}).set_index(['title', 'num'])
        result = wide_to_long(df, ['actor', 'actor_fb_likes'], i='title', j='num', sep='_')
        tm.assert_frame_equal(result, expected)

    def test_identical_stubnames(self):
        df = DataFrame({'A2010': [1.0, 2.0], 'A2011': [3.0, 4.0], 'B2010': [5.0, 6.0], 'A': ['X1', 'X2']})
        msg = "stubname can't be identical to a column name"
        with pytest.raises(ValueError, match=msg):
            wide_to_long(df, ['A', 'B'], i='A', j='colname')

    def test_nonnumeric_suffix(self):
        df = DataFrame({'treatment_placebo': [1.0, 2.0], 'treatment_test': [3.0, 4.0], 'result_placebo': [5.0, 6.0], 'A': ['X1', 'X2']})
        expected = DataFrame({'A': ['X1', 'X2', 'X1', 'X2'], 'colname': ['placebo', 'placebo', 'test', 'test'], 'result': [5.0, 6.0, np.nan, np.nan], 'treatment': [1.0, 2.0, 3.0, 4.0]})
        expected = expected.set_index(['A', 'colname'])
        result = wide_to_long(df, ['result', 'treatment'], i='A', j='colname', suffix='[a-z]+', sep='_')
        tm.assert_frame_equal(result, expected)

    def test_mixed_type_suffix(self):
        df = DataFrame({'A': ['X1', 'X2'], 'result_1': [0, 9], 'result_foo': [5.0, 6.0], 'treatment_1': [1.0, 2.0], 'treatment_foo': [3.0, 4.0]})
        expected = DataFrame({'A': ['X1', 'X2', 'X1', 'X2'], 'colname': ['1', '1', 'foo', 'foo'], 'result': [0.0, 9.0, 5.0, 6.0], 'treatment': [1.0, 2.0, 3.0, 4.0]}).set_index(['A', 'colname'])
        result = wide_to_long(df, ['result', 'treatment'], i='A', j='colname', suffix='.+', sep='_')
        tm.assert_frame_equal(result, expected)

    def test_float_suffix(self):
        df = DataFrame({'treatment_1.1': [1.0, 2.0], 'treatment_2.1': [3.0, 4.0], 'result_1.2': [5.0, 6.0], 'result_1': [0, 9], 'A': ['X1', 'X2']})
        expected = DataFrame({'A': ['X1', 'X2', 'X1', 'X2', 'X1', 'X2', 'X1', 'X2'], 'colname': [1.2, 1.2, 1.0, 1.0, 1.1, 1.1, 2.1, 2.1], 'result': [5.0, 6.0, 0.0, 9.0, np.nan, np.nan, np.nan, np.nan], 'treatment': [np.nan, np.nan, np.nan, np.nan, 1.0, 2.0, 3.0, 4.0]})
        expected = expected.set_index(['A', 'colname'])
        result = wide_to_long(df, ['result', 'treatment'], i='A', j='colname', suffix='[0-9.]+', sep='_')
        tm.assert_frame_equal(result, expected)

    def test_col_substring_of_stubname(self):
        wide_data = {'node_id': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}, 'A': {0: 0.8, 1: 0.0, 2: 0.25, 3: 1.0, 4: 0.81}, 'PA0': {0: 0.74, 1: 0.56, 2: 0.56, 3: 0.98, 4: 0.6}, 'PA1': {0: 0.77, 1: 0.64, 2: 0.52, 3: 0.98, 4: 0.67}, 'PA3': {0: 0.34, 1: 0.7, 2: 0.52, 3: 0.98, 4: 0.67}}
        wide_df = DataFrame.from_dict(wide_data)
        expected = wide_to_long(wide_df, stubnames=['PA'], i=['node_id', 'A'], j='time')
        result = wide_to_long(wide_df, stubnames='PA', i=['node_id', 'A'], j='time')
        tm.assert_frame_equal(result, expected)

    def test_raise_of_column_name_value(self):
        df = DataFrame({'col': list('ABC'), 'value': range(10, 16, 2)})
        with pytest.raises(ValueError, match=re.escape('value_name (value) cannot match')):
            df.melt(id_vars='value', value_name='value')

    @pytest.mark.parametrize('dtype', ['O', 'string'])
    def test_missing_stubname(self, dtype):
        df = DataFrame({'id': ['1', '2'], 'a-1': [100, 200], 'a-2': [300, 400]})
        df = df.astype({'id': dtype})
        result = wide_to_long(df, stubnames=['a', 'b'], i='id', j='num', sep='-')
        index = Index([('1', 1), ('2', 1), ('1', 2), ('2', 2)], name=('id', 'num'))
        expected = DataFrame({'a': [100, 200, 300, 400], 'b': [np.nan] * 4}, index=index)
        new_level = expected.index.levels[0].astype(dtype)
        expected.index = expected.index.set_levels(new_level, level=0)
        tm.assert_frame_equal(result, expected)