import datetime
from datetime import timedelta
from decimal import Decimal
from io import (
import json
import os
import sys
import time
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import IS64
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.json import ujson_dumps
def test_round_trip_exception(self, datapath):
    path = datapath('io', 'json', 'data', 'teams.csv')
    df = pd.read_csv(path)
    s = df.to_json()
    result = read_json(StringIO(s))
    res = result.reindex(index=df.index, columns=df.columns)
    msg = "The 'downcast' keyword in fillna is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = res.fillna(np.nan, downcast=False)
    tm.assert_frame_equal(res, df)