from collections.abc import Iterable
import datetime
import decimal
import hypothesis as h
import hypothesis.strategies as st
import itertools
import pytest
import struct
import subprocess
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
def test_safe_cast_nan_to_int_raises():
    arr = pa.array([np.nan, 1.0])
    with pytest.raises(pa.ArrowInvalid, match='truncated'):
        arr.cast(pa.int64(), safe=True)