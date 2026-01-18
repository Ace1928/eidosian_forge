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
@pytest.mark.parametrize('dtype', [False, None])
def test_series_roundtrip_object(self, orient, dtype, object_series):
    data = StringIO(object_series.to_json(orient=orient))
    result = read_json(data, typ='series', orient=orient, dtype=dtype)
    expected = object_series
    if orient in ('values', 'records'):
        expected = expected.reset_index(drop=True)
    if orient != 'split':
        expected.name = None
    tm.assert_series_equal(result, expected)