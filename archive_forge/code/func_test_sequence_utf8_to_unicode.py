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
@pytest.mark.parametrize('ty', [pa.string(), pa.large_string()])
def test_sequence_utf8_to_unicode(ty):
    data = [b'foo', None, b'bar']
    arr = pa.array(data, type=ty)
    assert arr.type == ty
    assert arr[0].as_py() == 'foo'
    val = 'mañana'.encode('utf-16-le')
    with pytest.raises(pa.ArrowInvalid):
        pa.array([val], type=ty)