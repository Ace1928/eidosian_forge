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
def test_table_from_pandas_schema_index_columns__unnamed_index():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3]})
    expected_schema = pa.schema([('a', pa.int64()), ('b', pa.float64()), ('__index_level_0__', pa.int64())])
    schema = pa.Schema.from_pandas(df, preserve_index=True)
    table = pa.Table.from_pandas(df, preserve_index=True, schema=schema)
    assert table.schema.remove_metadata().equals(expected_schema)
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3]}, index=[0, 1, 2])
    schema = pa.Schema.from_pandas(df)
    table = pa.Table.from_pandas(df, schema=schema)
    assert table.schema.remove_metadata().equals(expected_schema)