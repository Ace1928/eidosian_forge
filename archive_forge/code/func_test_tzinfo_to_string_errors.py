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
def test_tzinfo_to_string_errors():
    msg = 'Not an instance of datetime.tzinfo'
    with pytest.raises(TypeError):
        pa.lib.tzinfo_to_string('Europe/Budapest')
    if sys.version_info >= (3, 8):
        tz = datetime.timezone(datetime.timedelta(hours=1, seconds=30))
        msg = 'Offset must represent whole number of minutes'
        with pytest.raises(ValueError, match=msg):
            pa.lib.tzinfo_to_string(tz)