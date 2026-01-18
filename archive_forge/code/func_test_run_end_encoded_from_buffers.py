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
def test_run_end_encoded_from_buffers():
    run_ends = [3, 5, 10, 19]
    values = [1, 2, 1, 3]
    ree_type = pa.run_end_encoded(run_end_type=pa.int32(), value_type=pa.uint8())
    length = 19
    buffers = [None]
    null_count = 0
    offset = 0
    children = [run_ends, values]
    ree_array = pa.RunEndEncodedArray.from_buffers(ree_type, length, buffers, null_count, offset, children)
    check_run_end_encoded(ree_array, run_ends, values, 19, 4, 0)
    ree_array = pa.RunEndEncodedArray.from_buffers(ree_type, length, [], null_count, offset, children)
    check_run_end_encoded(ree_array, run_ends, values, 19, 4, 0)
    ree_array = pa.RunEndEncodedArray.from_buffers(ree_type, length, buffers, -1, offset, children)
    check_run_end_encoded(ree_array, run_ends, values, 19, 4, 0)
    ree_array = pa.RunEndEncodedArray.from_buffers(ree_type, length - 4, buffers, null_count, 4, children)
    check_run_end_encoded(ree_array, run_ends, values, length - 4, 3, 1)
    with pytest.raises(ValueError):
        pa.RunEndEncodedArray.from_buffers(ree_type, length, [None, None], null_count, offset, children)
    with pytest.raises(ValueError):
        pa.RunEndEncodedArray.from_buffers(ree_type, length, buffers, null_count, offset, None)
    with pytest.raises(ValueError):
        pa.RunEndEncodedArray.from_buffers(ree_type, length, buffers, null_count, offset, [run_ends])
    with pytest.raises(ValueError):
        pa.RunEndEncodedArray.from_buffers(ree_type, length, buffers, 1, offset, children)