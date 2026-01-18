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
def test_sequence_all_none(seq):
    arr = pa.array(seq([None, None]))
    assert len(arr) == 2
    assert arr.null_count == 2
    assert arr.type == pa.null()
    assert arr.to_pylist() == [None, None]