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
def test_timezone_string_roundtrip_pytz():
    pytz = pytest.importorskip('pytz')
    tz = [pytz.FixedOffset(90), pytz.FixedOffset(-90), pytz.utc, pytz.timezone('America/New_York')]
    name = ['+01:30', '-01:30', 'UTC', 'America/New_York']
    assert [pa.lib.tzinfo_to_string(i) for i in tz] == name
    assert [pa.lib.string_to_tzinfo(i) for i in name] == tz