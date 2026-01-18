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
@pytest.mark.parametrize('as_object', [True, False])
@pytest.mark.parametrize('date_format', ['iso', 'epoch'])
@pytest.mark.parametrize('timedelta_typ', [pd.Timedelta, timedelta])
def test_timedelta_to_json(self, as_object, date_format, timedelta_typ):
    data = [timedelta_typ(days=1), timedelta_typ(days=2), pd.NaT]
    if as_object:
        data.append('a')
    ser = Series(data, index=data)
    if date_format == 'iso':
        expected = '{"P1DT0H0M0S":"P1DT0H0M0S","P2DT0H0M0S":"P2DT0H0M0S","null":null}'
    else:
        expected = '{"86400000":86400000,"172800000":172800000,"null":null}'
    if as_object:
        expected = expected.replace('}', ',"a":"a"}')
    result = ser.to_json(date_format=date_format)
    assert result == expected