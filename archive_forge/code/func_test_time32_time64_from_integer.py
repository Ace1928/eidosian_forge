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
def test_time32_time64_from_integer():
    result = pa.array([1, 2, None], type=pa.time32('s'))
    expected = pa.array([datetime.time(second=1), datetime.time(second=2), None], type=pa.time32('s'))
    assert result.equals(expected)
    result = pa.array([1, 2, None], type=pa.time32('ms'))
    expected = pa.array([datetime.time(microsecond=1000), datetime.time(microsecond=2000), None], type=pa.time32('ms'))
    assert result.equals(expected)
    result = pa.array([1, 2, None], type=pa.time64('us'))
    expected = pa.array([datetime.time(microsecond=1), datetime.time(microsecond=2), None], type=pa.time64('us'))
    assert result.equals(expected)
    result = pa.array([1000, 2000, None], type=pa.time64('ns'))
    expected = pa.array([datetime.time(microsecond=1), datetime.time(microsecond=2), None], type=pa.time64('ns'))
    assert result.equals(expected)