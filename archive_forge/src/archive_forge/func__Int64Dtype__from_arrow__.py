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
def _Int64Dtype__from_arrow__(self, array):
    if isinstance(array, pa.Array):
        arr = array
    else:
        arr = array.chunk(0)
    buflist = arr.buffers()
    data = np.frombuffer(buflist[-1], dtype='int64')[arr.offset:arr.offset + len(arr)]
    bitmask = buflist[0]
    if bitmask is not None:
        mask = pa.BooleanArray.from_buffers(pa.bool_(), len(arr), [None, bitmask])
        mask = np.asarray(mask)
    else:
        mask = np.ones(len(arr), dtype=bool)
    int_arr = pd.arrays.IntegerArray(data.copy(), ~mask, copy=False)
    return int_arr