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
@parametrize_with_sequence_types
def test_sequence_integer_inferred(seq):
    expected = [1, None, 3, None]
    arr = pa.array(seq(expected))
    assert len(arr) == 4
    assert arr.null_count == 2
    assert arr.type == pa.int64()
    assert arr.to_pylist() == expected