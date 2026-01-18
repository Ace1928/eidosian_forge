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
def test_sequence_numpy_timestamp():
    data = [np.datetime64(datetime.datetime(2007, 7, 13, 1, 23, 34, 123456)), None, np.datetime64(datetime.datetime(2006, 1, 13, 12, 34, 56, 432539)), np.datetime64(datetime.datetime(2010, 8, 13, 5, 46, 57, 437699))]
    arr = pa.array(data)
    assert len(arr) == 4
    assert arr.type == pa.timestamp('us')
    assert arr.null_count == 1
    assert arr[0].as_py() == datetime.datetime(2007, 7, 13, 1, 23, 34, 123456)
    assert arr[1].as_py() is None
    assert arr[2].as_py() == datetime.datetime(2006, 1, 13, 12, 34, 56, 432539)
    assert arr[3].as_py() == datetime.datetime(2010, 8, 13, 5, 46, 57, 437699)