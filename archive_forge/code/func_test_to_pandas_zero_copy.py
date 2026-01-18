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
def test_to_pandas_zero_copy():
    import gc
    arr = pa.array(range(10))
    for i in range(10):
        series = arr.to_pandas()
        assert sys.getrefcount(series) == 2
        series = None
    assert sys.getrefcount(arr) == 2
    for i in range(10):
        arr = pa.array(range(10))
        series = arr.to_pandas()
        arr = None
        gc.collect()
        base_refcount = sys.getrefcount(series.values.base)
        assert base_refcount == 2
        series.sum()