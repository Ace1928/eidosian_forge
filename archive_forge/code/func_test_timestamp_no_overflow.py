import datetime
import decimal
import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.tests import util
def test_timestamp_no_overflow():
    pytest.importorskip('pytz')
    import pytz
    timestamps = [datetime.datetime(1, 1, 1, 0, 0, 0, tzinfo=pytz.utc), datetime.datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=pytz.utc), datetime.datetime(1970, 1, 1, 0, 0, 0, tzinfo=pytz.utc)]
    for ts in timestamps:
        s = pa.scalar(ts, type=pa.timestamp('us', tz='UTC'))
        assert s.as_py() == ts