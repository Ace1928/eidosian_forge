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
@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
@pytest.mark.parametrize('tz', [None, 'UTC'])
def test_to_numpy_datetime64(unit, tz):
    arr = pa.array([1, 2, 3], pa.timestamp(unit, tz=tz))
    expected = np.array([1, 2, 3], dtype='datetime64[{}]'.format(unit))
    np_arr = arr.to_numpy()
    np.testing.assert_array_equal(np_arr, expected)