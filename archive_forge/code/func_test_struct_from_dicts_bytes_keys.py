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
def test_struct_from_dicts_bytes_keys():
    ty = pa.struct([pa.field('a', pa.int32()), pa.field('b', pa.string()), pa.field('c', pa.bool_())])
    arr = pa.array([], type=ty)
    assert arr.to_pylist() == []
    data = [{b'a': 5, b'b': 'foo'}, {b'a': 6, b'c': False}]
    arr = pa.array(data, type=ty)
    assert arr.to_pylist() == [{'a': 5, 'b': 'foo', 'c': None}, {'a': 6, 'b': None, 'c': False}]