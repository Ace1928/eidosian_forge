import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
@pytest.mark.parametrize('indexer,expected_error,expected_error_msg', [(lambda s: s.__getitem__((2000, 3, 4)), KeyError, '^\\(2000, 3, 4\\)$'), (lambda s: s[2000, 3, 4], KeyError, '^\\(2000, 3, 4\\)$'), (lambda s: s.loc[2000, 3, 4], KeyError, '^\\(2000, 3, 4\\)$'), (lambda s: s.loc[2000, 3, 4, 5], IndexingError, 'Too many indexers'), (lambda s: s.__getitem__(len(s)), KeyError, ''), (lambda s: s[len(s)], KeyError, ''), (lambda s: s.iloc[len(s)], IndexError, 'single positional indexer is out-of-bounds')])
def test_series_getitem_indexing_errors(multiindex_year_month_day_dataframe_random_data, indexer, expected_error, expected_error_msg):
    s = multiindex_year_month_day_dataframe_random_data['A']
    with pytest.raises(expected_error, match=expected_error_msg):
        indexer(s)