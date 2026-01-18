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
def test_structarray():
    arr = pa.StructArray.from_arrays([], names=[])
    assert arr.type == pa.struct([])
    assert len(arr) == 0
    assert arr.to_pylist() == []
    ints = pa.array([None, 2, 3], type=pa.int64())
    strs = pa.array(['a', None, 'c'], type=pa.string())
    bools = pa.array([True, False, None], type=pa.bool_())
    arr = pa.StructArray.from_arrays([ints, strs, bools], ['ints', 'strs', 'bools'])
    expected = [{'ints': None, 'strs': 'a', 'bools': True}, {'ints': 2, 'strs': None, 'bools': False}, {'ints': 3, 'strs': 'c', 'bools': None}]
    pylist = arr.to_pylist()
    assert pylist == expected, (pylist, expected)
    with pytest.raises(ValueError):
        pa.StructArray.from_arrays([ints], ['ints', 'strs'])