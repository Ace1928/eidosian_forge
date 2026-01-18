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
def test_object_leak_in_dataframe():
    arr = pa.array([{'a': 1}])
    table = pa.table([arr], ['f0'])
    col = table.to_pandas()['f0']
    assert col.dtype == np.dtype('object')
    obj = col[0]
    refcount = sys.getrefcount(obj)
    assert sys.getrefcount(obj) == refcount
    del col
    assert sys.getrefcount(obj) == refcount - 1