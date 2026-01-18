from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_repr_to_string(self, multiindex_year_month_day_dataframe_random_data, multiindex_dataframe_random_data):
    ymd = multiindex_year_month_day_dataframe_random_data
    frame = multiindex_dataframe_random_data
    repr(frame)
    repr(ymd)
    repr(frame.T)
    repr(ymd.T)
    buf = StringIO()
    frame.to_string(buf=buf)
    ymd.to_string(buf=buf)
    frame.T.to_string(buf=buf)
    ymd.T.to_string(buf=buf)