import datetime
import decimal
import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.tests import util
def test_time_from_datetime_time():
    t1 = datetime.time(18, 0)
    t2 = datetime.time(21, 0)
    types = [pa.time32('s'), pa.time32('ms'), pa.time64('us'), pa.time64('ns')]
    for ty in types:
        for t in [t1, t2]:
            s = pa.scalar(t, type=ty)
            assert s.as_py() == t