import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
def test_series_getitem_returns_scalar(multiindex_year_month_day_dataframe_random_data, indexer_sl):
    s = multiindex_year_month_day_dataframe_random_data['A']
    expected = s.iloc[49]
    result = indexer_sl(s)[2000, 3, 10]
    assert result == expected