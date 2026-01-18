import gc
import decimal
import json
import multiprocessing as mp
import sys
import warnings
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from pyarrow.pandas_compat import get_logical_type, _pandas_api
from pyarrow.tests.util import invoke_script, random_ascii, rands
import pyarrow.tests.strategies as past
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
import pyarrow as pa
def test_timestamp_as_object_fixed_offset():
    pytz = pytest.importorskip('pytz')
    import datetime
    timezone = pytz.FixedOffset(120)
    dt = timezone.localize(datetime.datetime(2022, 5, 12, 16, 57))
    table = pa.table({'timestamp_col': pa.array([dt])})
    result = table.to_pandas(timestamp_as_object=True)
    assert pa.table(result) == table