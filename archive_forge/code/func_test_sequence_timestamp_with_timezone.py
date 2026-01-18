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
@pytest.mark.parametrize('timezone', [None, 'UTC', 'Etc/GMT-1', 'Europe/Budapest'])
@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
def test_sequence_timestamp_with_timezone(timezone, unit):
    pytz = pytest.importorskip('pytz')

    def expected_integer_value(dt):
        units = ['s', 'ms', 'us', 'ns']
        multiplier = 10 ** (units.index(unit) * 3)
        if dt is None:
            return None
        else:
            ts = decimal.Decimal(str(dt.timestamp()))
            return int(ts * multiplier)

    def expected_datetime_value(dt):
        if dt is None:
            return None
        if unit == 's':
            dt = dt.replace(microsecond=0)
        elif unit == 'ms':
            dt = dt.replace(microsecond=dt.microsecond // 1000 * 1000)
        if timezone is None:
            return dt.replace(tzinfo=None)
        else:
            return dt.astimezone(pytz.timezone(timezone))
    data = [datetime.datetime(2007, 7, 13, 8, 23, 34, 123456), pytz.utc.localize(datetime.datetime(2008, 1, 5, 5, 0, 0, 1000)), None, pytz.timezone('US/Eastern').localize(datetime.datetime(2006, 1, 13, 12, 34, 56, 432539)), pytz.timezone('Europe/Moscow').localize(datetime.datetime(2010, 8, 13, 5, 0, 0, 437699))]
    utcdata = [pytz.utc.localize(data[0]), data[1], None, data[3].astimezone(pytz.utc), data[4].astimezone(pytz.utc)]
    ty = pa.timestamp(unit, tz=timezone)
    arr = pa.array(data, type=ty)
    assert len(arr) == 5
    assert arr.type == ty
    assert arr.null_count == 1
    values = arr.cast('int64')
    expected = list(map(expected_integer_value, utcdata))
    assert values.to_pylist() == expected
    for i in range(len(arr)):
        assert arr[i].as_py() == expected_datetime_value(utcdata[i])