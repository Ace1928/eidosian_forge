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
def test_sequence_date():
    data = [datetime.date(2000, 1, 1), None, datetime.date(1970, 1, 1), datetime.date(2040, 2, 26)]
    arr = pa.array(data)
    assert len(arr) == 4
    assert arr.type == pa.date32()
    assert arr.null_count == 1
    assert arr[0].as_py() == datetime.date(2000, 1, 1)
    assert arr[1].as_py() is None
    assert arr[2].as_py() == datetime.date(1970, 1, 1)
    assert arr[3].as_py() == datetime.date(2040, 2, 26)