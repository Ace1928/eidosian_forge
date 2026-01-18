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
def test_struct_from_list_of_pairs():
    ty = pa.struct([pa.field('a', pa.int32()), pa.field('b', pa.string()), pa.field('c', pa.bool_())])
    data = [[('a', 5), ('b', 'foo'), ('c', True)], [('a', 6), ('b', 'bar'), ('c', False)], None]
    arr = pa.array(data, type=ty)
    assert arr.to_pylist() == [{'a': 5, 'b': 'foo', 'c': True}, {'a': 6, 'b': 'bar', 'c': False}, None]
    ty = pa.struct([pa.field('a', pa.int32()), pa.field('a', pa.string()), pa.field('b', pa.bool_())])
    data = [[('a', 5), ('a', 'foo'), ('b', True)], [('a', 6), ('a', 'bar'), ('b', False)]]
    arr = pa.array(data, type=ty)
    with pytest.raises(ValueError):
        arr.to_pylist()
    ty = pa.struct([pa.field('a', pa.int32()), pa.field('b', pa.string()), pa.field('c', pa.bool_())])
    data = [[], [('a', 5), ('b', 'foo'), ('c', True)], [('a', 2), ('b', 'baz')], [('a', 1), ('b', 'bar'), ('c', False), ('d', 'julia')]]
    expected = [{'a': None, 'b': None, 'c': None}, {'a': 5, 'b': 'foo', 'c': True}, {'a': 2, 'b': 'baz', 'c': None}, {'a': 1, 'b': 'bar', 'c': False}]
    arr = pa.array(data, type=ty)
    assert arr.to_pylist() == expected