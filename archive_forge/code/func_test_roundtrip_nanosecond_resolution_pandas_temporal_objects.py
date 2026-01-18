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
def test_roundtrip_nanosecond_resolution_pandas_temporal_objects():
    import pandas as pd
    ty = pa.duration('ns')
    arr = pa.array([9223371273709551616], type=ty)
    data = arr.to_pylist()
    assert isinstance(data[0], pd.Timedelta)
    restored = pa.array(data, type=ty)
    assert arr.equals(restored)
    assert restored.to_pylist() == [pd.Timedelta(9223371273709551616, unit='ns')]
    ty = pa.timestamp('ns')
    arr = pa.array([9223371273709551616], type=ty)
    data = arr.to_pylist()
    assert isinstance(data[0], pd.Timestamp)
    restored = pa.array(data, type=ty)
    assert arr.equals(restored)
    assert restored.to_pylist() == [pd.Timestamp(9223371273709551616, unit='ns')]
    ty = pa.timestamp('ns', tz='US/Eastern')
    value = 1604119893000000000
    arr = pa.array([value], type=ty)
    data = arr.to_pylist()
    assert isinstance(data[0], pd.Timestamp)
    restored = pa.array(data, type=ty)
    assert arr.equals(restored)
    assert restored.to_pylist() == [pd.Timestamp(value, unit='ns').tz_localize('UTC').tz_convert('US/Eastern')]