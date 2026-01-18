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
@pytest.mark.parametrize('idx', [pd.Index(range(4)), pd.date_range('2020-08-30', freq='d', periods=4)._with_freq(None), pd.date_range('2020-08-30', freq='d', periods=4, tz='US/Central')._with_freq(None), pd.MultiIndex.from_product([pd.date_range('2020-08-30', freq='d', periods=2, tz='US/Central'), ['x', 'y']])])
@pytest.mark.parametrize('vals', [{'floats': [1.1, 2.2, 3.3, 4.4]}, {'dates': pd.date_range('2020-08-30', freq='d', periods=4)}, {'timezones': pd.date_range('2020-08-30', freq='d', periods=4, tz='Europe/London')}])
def test_read_json_table_timezones_orient(self, idx, vals, recwarn):
    df = DataFrame(vals, index=idx)
    out = df.to_json(orient='table')
    result = pd.read_json(out, orient='table')
    tm.assert_frame_equal(df, result)