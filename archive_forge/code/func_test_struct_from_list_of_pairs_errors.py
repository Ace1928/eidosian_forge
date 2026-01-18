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
def test_struct_from_list_of_pairs_errors():
    ty = pa.struct([pa.field('a', pa.int32()), pa.field('b', pa.string()), pa.field('c', pa.bool_())])
    data = [[], [('a', 5), ('c', True), ('b', None)]]
    msg = 'The expected field name is `b` but `c` was given'
    with pytest.raises(ValueError, match=msg):
        pa.array(data, type=ty)
    template = 'Could not convert {} with type {}: was expecting tuple of (key, value) pair'
    cases = [tuple(), tuple('a'), tuple('unknown-key'), 'string']
    for key_value_pair in cases:
        msg = re.escape(template.format(repr(key_value_pair), type(key_value_pair).__name__))
        with pytest.raises(TypeError, match=msg):
            pa.array([[key_value_pair], [('a', 5), ('b', 'foo'), ('c', None)]], type=ty)
        with pytest.raises(TypeError, match=msg):
            pa.array([[('a', 5), ('b', 'foo'), ('c', None)], [key_value_pair]], type=ty)