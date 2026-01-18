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
@h.given(timezones)
def test_pytz_timezone_roundtrip(tz):
    if tz is None:
        pytest.skip('requires timezone not None')
    timezone_string = pa.lib.tzinfo_to_string(tz)
    timezone_tzinfo = pa.lib.string_to_tzinfo(timezone_string)
    assert timezone_tzinfo == tz