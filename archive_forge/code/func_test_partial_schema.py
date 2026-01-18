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
def test_partial_schema(self):
    data = OrderedDict([('a', [0, 1, 2, 3, 4]), ('b', np.array([-10, -5, 0, 5, 10], dtype=np.int32)), ('c', [-10, -5, 0, 5, 10])])
    df = pd.DataFrame(data)
    partial_schema = pa.schema([pa.field('c', pa.int64()), pa.field('a', pa.int64())])
    _check_pandas_roundtrip(df, schema=partial_schema, expected=df[['c', 'a']], expected_schema=partial_schema)