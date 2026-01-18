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
def test_sequence_duration_from_int_with_unit(unit):
    data = [5]
    ty = pa.duration(unit)
    arr = pa.array(data, type=ty)
    assert len(arr) == 1
    assert arr.type == ty
    assert arr[0].value == 5