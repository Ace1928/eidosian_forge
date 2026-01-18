import datetime
import decimal
import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.tests import util
@pytest.mark.parametrize(['value', 'time_type'], [(1, pa.time32('s')), (2 ** 30, pa.time32('s')), (None, pa.time32('s')), (1, pa.time32('ms')), (2 ** 30, pa.time32('ms')), (None, pa.time32('ms')), (1, pa.time64('us')), (2 ** 62, pa.time64('us')), (None, pa.time64('us')), (1, pa.time64('ns')), (2 ** 62, pa.time64('ns')), (None, pa.time64('ns')), (1, pa.date32()), (2 ** 30, pa.date32()), (None, pa.date32()), (1, pa.date64()), (2 ** 62, pa.date64()), (None, pa.date64()), (1, pa.timestamp('ns')), (2 ** 62, pa.timestamp('ns')), (None, pa.timestamp('ns')), (1, pa.duration('ns')), (2 ** 62, pa.duration('ns')), (None, pa.duration('ns')), ((1, 2, -3), pa.month_day_nano_interval()), (None, pa.month_day_nano_interval())])
def test_temporal_values(value, time_type: pa.DataType):
    time_scalar = pa.scalar(value, type=time_type)
    time_scalar.validate(full=True)
    assert time_scalar.value == value