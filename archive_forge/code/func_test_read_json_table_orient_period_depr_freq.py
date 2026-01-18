from collections import OrderedDict
from io import StringIO
import json
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.json._table_schema import (
@pytest.mark.parametrize('freq', ['M', '2M', 'Q', '2Q', 'Y', '2Y'])
def test_read_json_table_orient_period_depr_freq(self, freq, recwarn):
    df = DataFrame({'ints': [1, 2]}, index=pd.PeriodIndex(['2020-01', '2021-06'], freq=freq))
    out = df.to_json(orient='table')
    result = pd.read_json(out, orient='table')
    tm.assert_frame_equal(df, result)