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
@pytest.mark.parametrize('timedelta_typ', [pd.Timedelta, timedelta])
def test_timedelta_to_json_fractional_precision(self, as_object, timedelta_typ):
    data = [timedelta_typ(milliseconds=42)]
    ser = Series(data, index=data)
    if as_object:
        ser = ser.astype(object)
    result = ser.to_json()
    expected = '{"42":42}'
    assert result == expected