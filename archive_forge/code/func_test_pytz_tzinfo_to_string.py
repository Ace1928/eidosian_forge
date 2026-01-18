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
def test_pytz_tzinfo_to_string():
    pytz = pytest.importorskip('pytz')
    tz = [pytz.utc, pytz.timezone('Europe/Paris')]
    expected = ['UTC', 'Europe/Paris']
    assert [pa.lib.tzinfo_to_string(i) for i in tz] == expected
    tz = [pytz.timezone('Etc/GMT-9'), pytz.FixedOffset(180)]
    expected = ['Etc/GMT-9', '+03:00']
    assert [pa.lib.tzinfo_to_string(i) for i in tz] == expected