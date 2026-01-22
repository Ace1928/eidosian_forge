import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
class TestSortValuesLevelAsStr:

    def test_sort_index_level_and_column_label(self, df_none, df_idx, sort_names, ascending, request):
        if Version(np.__version__) >= Version('1.25') and request.node.callspec.id == 'df_idx0-inner-True':
            request.applymarker(pytest.mark.xfail(reason='pandas default unstable sorting of duplicatesissue with numpy>=1.25 with AVX instructions', strict=False))
        levels = df_idx.index.names
        expected = df_none.sort_values(by=sort_names, ascending=ascending, axis=0).set_index(levels)
        result = df_idx.sort_values(by=sort_names, ascending=ascending, axis=0)
        tm.assert_frame_equal(result, expected)

    def test_sort_column_level_and_index_label(self, df_none, df_idx, sort_names, ascending, request):
        levels = df_idx.index.names
        expected = df_none.sort_values(by=sort_names, ascending=ascending, axis=0).set_index(levels).T
        result = df_idx.T.sort_values(by=sort_names, ascending=ascending, axis=1)
        if Version(np.__version__) >= Version('1.25'):
            request.applymarker(pytest.mark.xfail(reason='pandas default unstable sorting of duplicatesissue with numpy>=1.25 with AVX instructions', strict=False))
        tm.assert_frame_equal(result, expected)

    def test_sort_values_validate_ascending_for_value_error(self):
        df = DataFrame({'D': [23, 7, 21]})
        msg = 'For argument "ascending" expected type bool, received type str.'
        with pytest.raises(ValueError, match=msg):
            df.sort_values(by='D', ascending='False')

    @pytest.mark.parametrize('ascending', [False, 0, 1, True])
    def test_sort_values_validate_ascending_functional(self, ascending):
        df = DataFrame({'D': [23, 7, 21]})
        indexer = df['D'].argsort().values
        if not ascending:
            indexer = indexer[::-1]
        expected = df.loc[df.index[indexer]]
        result = df.sort_values(by='D', ascending=ascending)
        tm.assert_frame_equal(result, expected)