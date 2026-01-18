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
def test_to_numpy_zero_copy():
    arr = pa.array(range(10))
    np_arr = arr.to_numpy()
    arrow_buf = arr.buffers()[1]
    assert arrow_buf.address == np_arr.ctypes.data
    arr = None
    import gc
    gc.collect()
    assert np_arr.base is not None
    expected = np.arange(10)
    np.testing.assert_array_equal(np_arr, expected)