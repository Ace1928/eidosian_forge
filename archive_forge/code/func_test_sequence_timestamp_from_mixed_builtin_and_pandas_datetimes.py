import collections
import datetime
import decimal
import itertools
import math
import re
import sys
import hypothesis as h
import numpy as np
import pytest
from pyarrow.pandas_compat import _pandas_api  # noqa
import pyarrow as pa
from pyarrow.tests import util
import pyarrow.tests.strategies as past
@pytest.mark.pandas
def test_sequence_timestamp_from_mixed_builtin_and_pandas_datetimes():
    pytest.importorskip('pytz')
    import pytz
    import pandas as pd
    data = [pd.Timestamp(1184307814123456123, tz=pytz.timezone('US/Eastern'), unit='ns'), datetime.datetime(2007, 7, 13, 8, 23, 34, 123456), pytz.utc.localize(datetime.datetime(2008, 1, 5, 5, 0, 0, 1000)), None]
    utcdata = [data[0].astimezone(pytz.utc), pytz.utc.localize(data[1]), data[2].astimezone(pytz.utc), None]
    arr = pa.array(data)
    assert arr.type == pa.timestamp('us', tz='US/Eastern')
    values = arr.cast('int64')
    expected = [int(dt.timestamp() * 10 ** 6) if dt else None for dt in utcdata]
    assert values.to_pylist() == expected