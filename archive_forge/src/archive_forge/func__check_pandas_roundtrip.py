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
def _check_pandas_roundtrip(df, expected=None, use_threads=False, expected_schema=None, check_dtype=True, schema=None, preserve_index=False, as_batch=False):
    klass = pa.RecordBatch if as_batch else pa.Table
    table = klass.from_pandas(df, schema=schema, preserve_index=preserve_index, nthreads=2 if use_threads else 1)
    result = table.to_pandas(use_threads=use_threads)
    if expected_schema:
        assert table.schema.equals(expected_schema)
    if expected is None:
        expected = df
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'elementwise comparison failed', DeprecationWarning)
        tm.assert_frame_equal(result, expected, check_dtype=check_dtype, check_index_type='equiv' if preserve_index else False)