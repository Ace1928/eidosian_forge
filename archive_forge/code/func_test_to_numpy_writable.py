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
def test_to_numpy_writable():
    arr = pa.array(range(10))
    np_arr = arr.to_numpy()
    with pytest.raises(ValueError):
        np_arr[0] = 10
    np_arr2 = arr.to_numpy(zero_copy_only=False, writable=True)
    np_arr2[0] = 10
    assert arr[0].as_py() == 0
    with pytest.raises(ValueError):
        arr.to_numpy(zero_copy_only=True, writable=True)