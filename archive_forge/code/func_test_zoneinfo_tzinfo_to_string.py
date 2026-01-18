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
def test_zoneinfo_tzinfo_to_string():
    zoneinfo = pytest.importorskip('zoneinfo')
    if sys.platform == 'win32':
        pytest.importorskip('tzdata')
    tz = zoneinfo.ZoneInfo('UTC')
    assert pa.lib.tzinfo_to_string(tz) == 'UTC'
    tz = zoneinfo.ZoneInfo('Europe/Paris')
    assert pa.lib.tzinfo_to_string(tz) == 'Europe/Paris'