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
@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
def test_sequence_duration_with_unit(unit):
    data = [datetime.timedelta(3, 22, 1001)]
    expected = {'s': datetime.timedelta(3, 22), 'ms': datetime.timedelta(3, 22, 1000), 'us': datetime.timedelta(3, 22, 1001), 'ns': datetime.timedelta(3, 22, 1001)}
    ty = pa.duration(unit)
    arr_s = pa.array(data, type=ty)
    assert len(arr_s) == 1
    assert arr_s.type == ty
    assert arr_s[0].as_py() == expected[unit]