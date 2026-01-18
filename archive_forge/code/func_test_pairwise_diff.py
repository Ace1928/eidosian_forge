from collections import namedtuple
import datetime
import decimal
from functools import lru_cache, partial
import inspect
import itertools
import math
import os
import pytest
import random
import sys
import textwrap
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import ArrowNotImplementedError
from pyarrow.tests import util
def test_pairwise_diff():
    arr = pa.array([1, 2, 3, None, 4, 5])
    expected = pa.array([None, 1, 1, None, None, 1])
    result = pa.compute.pairwise_diff(arr, period=1)
    assert result.equals(expected)
    arr = pa.array([1, 2, 3, None, 4, 5])
    expected = pa.array([None, None, 2, None, 1, None])
    result = pa.compute.pairwise_diff(arr, period=2)
    assert result.equals(expected)
    arr = pa.array([1, 2, 3, None, 4, 5], type=pa.int8())
    expected = pa.array([-1, -1, None, None, -1, None], type=pa.int8())
    result = pa.compute.pairwise_diff(arr, period=-1)
    assert result.equals(expected)
    arr = pa.array([1, 2, 3, None, 4, 5], type=pa.uint8())
    expected = pa.array([255, 255, None, None, 255, None], type=pa.uint8())
    result = pa.compute.pairwise_diff(arr, period=-1)
    assert result.equals(expected)
    arr = pa.array([1, 2, 3, None, 4, 5], type=pa.uint8())
    with pytest.raises(pa.ArrowInvalid, match='overflow'):
        pa.compute.pairwise_diff_checked(arr, period=-1)