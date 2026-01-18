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
def test_table_uses_memory_pool():
    N = 10000
    arr = pa.array(np.arange(N, dtype=np.int64))
    t = pa.table([arr, arr, arr], ['f0', 'f1', 'f2'])
    prior_allocation = pa.total_allocated_bytes()
    x = t.to_pandas()
    assert pa.total_allocated_bytes() == prior_allocation + 3 * N * 8
    x = None
    gc.collect()
    assert pa.total_allocated_bytes() == prior_allocation