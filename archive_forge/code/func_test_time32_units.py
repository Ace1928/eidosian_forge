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
def test_time32_units():
    for valid_unit in ('s', 'ms'):
        ty = pa.time32(valid_unit)
        assert ty.unit == valid_unit
    for invalid_unit in ('m', 'us', 'ns'):
        error_msg = 'Invalid time unit for time32: {!r}'.format(invalid_unit)
        with pytest.raises(ValueError, match=error_msg):
            pa.time32(invalid_unit)