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
@pytest.mark.xfail(reason='Type inference for uint64 not implemented', raises=OverflowError)
def test_uint64_max_convert():
    data = [0, np.iinfo(np.uint64).max]
    arr = pa.array(data, type=pa.uint64())
    expected = pa.array(np.array(data, dtype='uint64'))
    assert arr.equals(expected)
    arr_inferred = pa.array(data)
    assert arr_inferred.equals(expected)