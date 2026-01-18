from collections.abc import Iterable
import datetime
import decimal
import hypothesis as h
import hypothesis.strategies as st
import itertools
import pytest
import struct
import subprocess
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
@pytest.mark.pandas
def test_to_pandas_timezone():
    arr = pa.array([1, 2, 3], type=pa.timestamp('s', tz='Europe/Brussels'))
    s = arr.to_pandas()
    assert s.dt.tz is not None
    arr = pa.chunked_array([arr])
    s = arr.to_pandas()
    assert s.dt.tz is not None