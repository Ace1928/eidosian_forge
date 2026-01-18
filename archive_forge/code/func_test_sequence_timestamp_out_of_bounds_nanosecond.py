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
def test_sequence_timestamp_out_of_bounds_nanosecond():
    data = [datetime.datetime(2262, 4, 12)]
    with pytest.raises(ValueError, match='out of bounds'):
        pa.array(data, type=pa.timestamp('ns'))
    arr = pa.array(data, type=pa.timestamp('us'))
    assert arr.to_pylist() == data
    tz = datetime.timezone(datetime.timedelta(hours=-1))
    data = [datetime.datetime(2262, 4, 11, 23, tzinfo=tz)]
    with pytest.raises(ValueError, match='out of bounds'):
        pa.array(data, type=pa.timestamp('ns'))
    arr = pa.array(data, type=pa.timestamp('us'))
    assert arr.to_pylist()[0] == datetime.datetime(2262, 4, 12)