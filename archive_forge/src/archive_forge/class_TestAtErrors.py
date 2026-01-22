from datetime import (
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
class TestAtErrors:

    def test_at_series_raises_key_error(self, indexer_al):
        ser = Series([1, 2, 3], index=[3, 2, 1])
        result = indexer_al(ser)[1]
        assert result == 3
        with pytest.raises(KeyError, match='a'):
            indexer_al(ser)['a']

    def test_at_frame_raises_key_error(self, indexer_al):
        df = DataFrame({0: [1, 2, 3]}, index=[3, 2, 1])
        result = indexer_al(df)[1, 0]
        assert result == 3
        with pytest.raises(KeyError, match='a'):
            indexer_al(df)['a', 0]
        with pytest.raises(KeyError, match='a'):
            indexer_al(df)[1, 'a']

    def test_at_series_raises_key_error2(self, indexer_al):
        ser = Series([1, 2, 3], index=list('abc'))
        result = indexer_al(ser)['a']
        assert result == 1
        with pytest.raises(KeyError, match='^0$'):
            indexer_al(ser)[0]

    def test_at_frame_raises_key_error2(self, indexer_al):
        df = DataFrame({'A': [1, 2, 3]}, index=list('abc'))
        result = indexer_al(df)['a', 'A']
        assert result == 1
        with pytest.raises(KeyError, match='^0$'):
            indexer_al(df)['a', 0]

    def test_at_frame_multiple_columns(self):
        df = DataFrame({'a': [1, 2], 'b': [3, 4]})
        new_row = [6, 7]
        with pytest.raises(InvalidIndexError, match=f'You can only assign a scalar value not a \\{type(new_row)}'):
            df.at[5] = new_row

    def test_at_getitem_mixed_index_no_fallback(self):
        ser = Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 1, 2])
        with pytest.raises(KeyError, match='^0$'):
            ser.at[0]
        with pytest.raises(KeyError, match='^4$'):
            ser.at[4]

    def test_at_categorical_integers(self):
        ci = CategoricalIndex([3, 4])
        arr = np.arange(4).reshape(2, 2)
        frame = DataFrame(arr, index=ci)
        for df in [frame, frame.T]:
            for key in [0, 1]:
                with pytest.raises(KeyError, match=str(key)):
                    df.at[key, key]

    def test_at_applied_for_rows(self):
        df = DataFrame(index=['a'], columns=['col1', 'col2'])
        new_row = [123, 15]
        with pytest.raises(InvalidIndexError, match=f'You can only assign a scalar value not a \\{type(new_row)}'):
            df.at['a'] = new_row