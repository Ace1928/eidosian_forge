import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reindex_preserve_levels(self, multiindex_year_month_day_dataframe_random_data, using_copy_on_write):
    ymd = multiindex_year_month_day_dataframe_random_data
    new_index = ymd.index[::10]
    chunk = ymd.reindex(new_index)
    if using_copy_on_write:
        assert chunk.index.is_(new_index)
    else:
        assert chunk.index is new_index
    chunk = ymd.loc[new_index]
    assert chunk.index.equals(new_index)
    ymdT = ymd.T
    chunk = ymdT.reindex(columns=new_index)
    if using_copy_on_write:
        assert chunk.columns.is_(new_index)
    else:
        assert chunk.columns is new_index
    chunk = ymdT.loc[:, new_index]
    assert chunk.columns.equals(new_index)