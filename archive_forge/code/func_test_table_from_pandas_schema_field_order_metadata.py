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
def test_table_from_pandas_schema_field_order_metadata():
    df = pd.DataFrame({'datetime': pd.date_range('2020-01-01T00:00:00Z', freq='H', periods=2), 'float': np.random.randn(2)})
    schema = pa.schema([pa.field('float', pa.float32(), nullable=True), pa.field('datetime', pa.timestamp('s', tz='UTC'), nullable=False)])
    table = pa.Table.from_pandas(df, schema=schema)
    assert table.schema.equals(schema)
    metadata_float = table.schema.pandas_metadata['columns'][0]
    assert metadata_float['name'] == 'float'
    assert metadata_float['metadata'] is None
    metadata_datetime = table.schema.pandas_metadata['columns'][1]
    assert metadata_datetime['name'] == 'datetime'
    assert metadata_datetime['metadata'] == {'timezone': 'UTC'}
    result = table.to_pandas()
    coerce_cols_to_types = {'float': 'float32'}
    if Version(pd.__version__) >= Version('2.0.0'):
        coerce_cols_to_types['datetime'] = 'datetime64[s, UTC]'
    expected = df[['float', 'datetime']].astype(coerce_cols_to_types)
    tm.assert_frame_equal(result, expected)