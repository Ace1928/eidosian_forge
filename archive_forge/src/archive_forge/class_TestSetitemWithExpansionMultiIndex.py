import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestSetitemWithExpansionMultiIndex:

    def test_setitem_new_column_mixed_depth(self):
        arrays = [['a', 'top', 'top', 'routine1', 'routine1', 'routine2'], ['', 'OD', 'OD', 'result1', 'result2', 'result1'], ['', 'wx', 'wy', '', '', '']]
        tuples = sorted(zip(*arrays))
        index = MultiIndex.from_tuples(tuples)
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 6)), columns=index)
        result = df.copy()
        expected = df.copy()
        result['b'] = [1, 2, 3, 4]
        expected['b', '', ''] = [1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

    def test_setitem_new_column_all_na(self):
        mix = MultiIndex.from_tuples([('1a', '2a'), ('1a', '2b'), ('1a', '2c')])
        df = DataFrame([[1, 2], [3, 4], [5, 6]], index=mix)
        s = Series({(1, 1): 1, (1, 2): 2})
        df['new'] = s
        assert df['new'].isna().all()

    def test_setitem_enlargement_keep_index_names(self):
        mi = MultiIndex.from_tuples([(1, 2, 3)], names=['i1', 'i2', 'i3'])
        df = DataFrame(data=[[10, 20, 30]], index=mi, columns=['A', 'B', 'C'])
        df.loc[0, 0, 0] = df.loc[1, 2, 3]
        mi_expected = MultiIndex.from_tuples([(1, 2, 3), (0, 0, 0)], names=['i1', 'i2', 'i3'])
        expected = DataFrame(data=[[10, 20, 30], [10, 20, 30]], index=mi_expected, columns=['A', 'B', 'C'])
        tm.assert_frame_equal(df, expected)