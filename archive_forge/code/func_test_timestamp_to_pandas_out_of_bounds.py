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
def test_timestamp_to_pandas_out_of_bounds(self):
    for unit in ['s', 'ms', 'us']:
        for tz in [None, 'America/New_York']:
            arr = pa.array([datetime(1, 1, 1)], pa.timestamp(unit, tz=tz))
            table = pa.table({'a': arr})
            msg = 'would result in out of bounds timestamp'
            with pytest.raises(ValueError, match=msg):
                arr.to_pandas(coerce_temporal_nanoseconds=True)
            with pytest.raises(ValueError, match=msg):
                table.to_pandas(coerce_temporal_nanoseconds=True)
            with pytest.raises(ValueError, match=msg):
                table.column('a').to_pandas(coerce_temporal_nanoseconds=True)
            arr.to_pandas(safe=False, coerce_temporal_nanoseconds=True)
            table.to_pandas(safe=False, coerce_temporal_nanoseconds=True)
            table.column('a').to_pandas(safe=False, coerce_temporal_nanoseconds=True)