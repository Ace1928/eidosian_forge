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
def test_sequence_double():
    data = [1.5, 1.0, None, 2.5, None, None]
    arr = pa.array(data)
    assert len(arr) == 6
    assert arr.null_count == 3
    assert arr.type == pa.float64()
    assert arr.to_pylist() == data