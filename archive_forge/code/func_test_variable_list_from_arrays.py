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
def test_variable_list_from_arrays():
    values = pa.array([1, 2, 3, 4], pa.int64())
    offsets = pa.array([0, 2, 4])
    result = pa.ListArray.from_arrays(offsets, values)
    assert result.to_pylist() == [[1, 2], [3, 4]]
    assert result.type.equals(pa.list_(pa.int64()))
    offsets = pa.array([0, None, 2, 4])
    result = pa.ListArray.from_arrays(offsets, values)
    assert result.to_pylist() == [[1, 2], None, [3, 4]]
    with pytest.raises(ValueError):
        pa.ListArray.from_arrays(pa.array([-1, 2, 4]), values)
    with pytest.raises(ValueError):
        pa.ListArray.from_arrays(pa.array([0, 2, 5]), values)