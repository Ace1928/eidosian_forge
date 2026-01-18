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
def test_sequence_fixed_size_bytes():
    data = [b'foof', None, bytearray(b'barb'), b'2346']
    arr = pa.array(data, type=pa.binary(4))
    assert len(arr) == 4
    assert arr.null_count == 1
    assert arr.type == pa.binary(4)
    assert arr.to_pylist() == [b'foof', None, b'barb', b'2346']