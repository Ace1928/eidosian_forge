from collections import OrderedDict
from collections.abc import Iterator
from functools import partial
import datetime
import sys
import pytest
import hypothesis as h
import hypothesis.strategies as st
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.types as types
import pyarrow.tests.strategies as past
def test_timedelta_overflow():
    d = datetime.timedelta(days=-106751992, seconds=71945, microseconds=224192)
    with pytest.raises(pa.ArrowInvalid):
        pa.scalar(d)
    d = datetime.timedelta(days=106751991, seconds=14454, microseconds=775808)
    with pytest.raises(pa.ArrowInvalid):
        pa.scalar(d)
    d = datetime.timedelta(days=-106752, seconds=763, microseconds=145224)
    with pytest.raises(pa.ArrowInvalid):
        pa.scalar(d, type=pa.duration('ns'))
    pa.scalar(d, type=pa.duration('us')).as_py() == d
    for d in [datetime.timedelta.min, datetime.timedelta.max]:
        pa.scalar(d, type=pa.duration('ms')).as_py() == d
        pa.scalar(d, type=pa.duration('s')).as_py() == d