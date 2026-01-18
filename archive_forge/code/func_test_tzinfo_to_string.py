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
@pytest.mark.parametrize(('tz', 'expected'), [(datetime.timezone.utc, 'UTC'), (datetime.timezone(datetime.timedelta(hours=1, minutes=30)), '+01:30')])
def test_tzinfo_to_string(tz, expected):
    assert pa.lib.tzinfo_to_string(tz) == expected