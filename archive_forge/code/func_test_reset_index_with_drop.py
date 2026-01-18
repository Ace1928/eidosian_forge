from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reset_index_with_drop(self, multiindex_year_month_day_dataframe_random_data):
    ymd = multiindex_year_month_day_dataframe_random_data
    deleveled = ymd.reset_index(drop=True)
    assert len(deleveled.columns) == len(ymd.columns)
    assert deleveled.index.name == ymd.index.name