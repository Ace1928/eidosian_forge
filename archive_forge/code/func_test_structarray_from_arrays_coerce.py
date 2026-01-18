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
def test_structarray_from_arrays_coerce():
    ints = [None, 2, 3]
    strs = ['a', None, 'c']
    bools = [True, False, None]
    ints_nonnull = [1, 2, 3]
    arrays = [ints, strs, bools, ints_nonnull]
    result = pa.StructArray.from_arrays(arrays, ['ints', 'strs', 'bools', 'int_nonnull'])
    expected = pa.StructArray.from_arrays([pa.array(ints, type='int64'), pa.array(strs, type='utf8'), pa.array(bools), pa.array(ints_nonnull, type='int64')], ['ints', 'strs', 'bools', 'int_nonnull'])
    with pytest.raises(ValueError):
        pa.StructArray.from_arrays(arrays)
    assert result.equals(expected)