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
@pytest.mark.parametrize('index_nm', [None, 'idx', pytest.param('index', marks=pytest.mark.xfail), 'level_0'])
@pytest.mark.parametrize('vals', [{'ints': [1, 2, 3, 4]}, {'objects': ['a', 'b', 'c', 'd']}, {'objects': ['1', '2', '3', '4']}, {'date_ranges': pd.date_range('2016-01-01', freq='d', periods=4)}, {'categoricals': pd.Series(pd.Categorical(['a', 'b', 'c', 'c']))}, {'ordered_cats': pd.Series(pd.Categorical(['a', 'b', 'c', 'c'], ordered=True))}, {'floats': [1.0, 2.0, 3.0, 4.0]}, {'floats': [1.1, 2.2, 3.3, 4.4]}, {'bools': [True, False, False, True]}, {'timezones': pd.date_range('2016-01-01', freq='d', periods=4, tz='US/Central')}])
def test_read_json_table_period_orient(self, index_nm, vals, recwarn):
    df = DataFrame(vals, index=pd.Index((pd.Period(f'2022Q{q}') for q in range(1, 5)), name=index_nm))
    out = df.to_json(orient='table')
    result = pd.read_json(out, orient='table')
    tm.assert_frame_equal(df, result)