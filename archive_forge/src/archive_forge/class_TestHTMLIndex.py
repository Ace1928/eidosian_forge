from datetime import datetime
from io import StringIO
import itertools
import re
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
class TestHTMLIndex:

    @pytest.fixture
    def df(self):
        index = ['foo', 'bar', 'baz']
        df = DataFrame({'A': [1, 2, 3], 'B': [1.2, 3.4, 5.6], 'C': ['one', 'two', np.nan]}, columns=['A', 'B', 'C'], index=index)
        return df

    @pytest.fixture
    def expected_without_index(self, datapath):
        return expected_html(datapath, 'index_2')

    def test_to_html_flat_index_without_name(self, datapath, df, expected_without_index):
        expected_with_index = expected_html(datapath, 'index_1')
        assert df.to_html() == expected_with_index
        result = df.to_html(index=False)
        for i in df.index:
            assert i not in result
        assert result == expected_without_index

    def test_to_html_flat_index_with_name(self, datapath, df, expected_without_index):
        df.index = Index(['foo', 'bar', 'baz'], name='idx')
        expected_with_index = expected_html(datapath, 'index_3')
        assert df.to_html() == expected_with_index
        assert df.to_html(index=False) == expected_without_index

    def test_to_html_multiindex_without_names(self, datapath, df, expected_without_index):
        tuples = [('foo', 'car'), ('foo', 'bike'), ('bar', 'car')]
        df.index = MultiIndex.from_tuples(tuples)
        expected_with_index = expected_html(datapath, 'index_4')
        assert df.to_html() == expected_with_index
        result = df.to_html(index=False)
        for i in ['foo', 'bar', 'car', 'bike']:
            assert i not in result
        assert result == expected_without_index

    def test_to_html_multiindex_with_names(self, datapath, df, expected_without_index):
        tuples = [('foo', 'car'), ('foo', 'bike'), ('bar', 'car')]
        df.index = MultiIndex.from_tuples(tuples, names=['idx1', 'idx2'])
        expected_with_index = expected_html(datapath, 'index_5')
        assert df.to_html() == expected_with_index
        assert df.to_html(index=False) == expected_without_index