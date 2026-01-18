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
def test_struct_from_tuples():
    ty = pa.struct([pa.field('a', pa.int32()), pa.field('b', pa.string()), pa.field('c', pa.bool_())])
    data = [(5, 'foo', True), (6, 'bar', False)]
    expected = [{'a': 5, 'b': 'foo', 'c': True}, {'a': 6, 'b': 'bar', 'c': False}]
    arr = pa.array(data, type=ty)
    data_as_ndarray = np.empty(len(data), dtype=object)
    data_as_ndarray[:] = data
    arr2 = pa.array(data_as_ndarray, type=ty)
    assert arr.to_pylist() == expected
    assert arr.equals(arr2)
    data = [(5, 'foo', None), None, (6, None, False)]
    expected = [{'a': 5, 'b': 'foo', 'c': None}, None, {'a': 6, 'b': None, 'c': False}]
    arr = pa.array(data, type=ty)
    assert arr.to_pylist() == expected
    for tup in [(5, 'foo'), (), ('5', 'foo', True, None)]:
        with pytest.raises(ValueError, match='(?i)tuple size'):
            pa.array([tup], type=ty)