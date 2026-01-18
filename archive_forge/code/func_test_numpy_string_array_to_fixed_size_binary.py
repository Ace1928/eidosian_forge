import gc
import decimal
import json
import multiprocessing as mp
import sys
import warnings
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from pyarrow.pandas_compat import get_logical_type, _pandas_api
from pyarrow.tests.util import invoke_script, random_ascii, rands
import pyarrow.tests.strategies as past
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
import pyarrow as pa
def test_numpy_string_array_to_fixed_size_binary(self):
    arr = np.array([b'foo', b'bar', b'baz'], dtype='|S3')
    converted = pa.array(arr, type=pa.binary(3))
    expected = pa.array(list(arr), type=pa.binary(3))
    assert converted.equals(expected)
    mask = np.array([False, True, False])
    converted = pa.array(arr, type=pa.binary(3), mask=mask)
    expected = pa.array([b'foo', None, b'baz'], type=pa.binary(3))
    assert converted.equals(expected)
    with pytest.raises(pa.lib.ArrowInvalid, match='Got bytestring of length 3 \\(expected 4\\)'):
        arr = np.array([b'foo', b'bar', b'baz'], dtype='|S3')
        pa.array(arr, type=pa.binary(4))
    with pytest.raises(pa.lib.ArrowInvalid, match='Got bytestring of length 12 \\(expected 3\\)'):
        arr = np.array([b'foo', b'bar', b'baz'], dtype='|U3')
        pa.array(arr, type=pa.binary(3))