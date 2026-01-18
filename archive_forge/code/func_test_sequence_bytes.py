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
def test_sequence_bytes():
    u1 = b'ma\xc3\xb1ana'
    data = [b'foo', memoryview(b'dada'), memoryview(b'd-a-t-a')[::2], u1.decode('utf-8'), bytearray(b'bar'), None]
    for ty in [None, pa.binary(), pa.large_binary()]:
        arr = pa.array(data, type=ty)
        assert len(arr) == 6
        assert arr.null_count == 1
        assert arr.type == ty or pa.binary()
        assert arr.to_pylist() == [b'foo', b'dada', b'data', u1, b'bar', None]