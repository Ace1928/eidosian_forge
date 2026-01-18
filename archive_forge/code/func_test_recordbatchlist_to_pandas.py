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
def test_recordbatchlist_to_pandas():
    data1 = pd.DataFrame({'c1': np.array([1, 1, 2], dtype='uint32'), 'c2': np.array([1.0, 2.0, 3.0], dtype='float64'), 'c3': [True, None, False], 'c4': ['foo', 'bar', None]})
    data2 = pd.DataFrame({'c1': np.array([3, 5], dtype='uint32'), 'c2': np.array([4.0, 5.0], dtype='float64'), 'c3': [True, True], 'c4': ['baz', 'qux']})
    batch1 = pa.RecordBatch.from_pandas(data1)
    batch2 = pa.RecordBatch.from_pandas(data2)
    table = pa.Table.from_batches([batch1, batch2])
    result = table.to_pandas()
    data = pd.concat([data1, data2]).reset_index(drop=True)
    tm.assert_frame_equal(data, result)