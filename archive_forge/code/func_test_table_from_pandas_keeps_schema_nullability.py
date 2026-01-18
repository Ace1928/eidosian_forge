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
def test_table_from_pandas_keeps_schema_nullability():
    df = pd.DataFrame({'a': [1, 2, 3, 4]})
    schema = pa.schema([pa.field('a', pa.int64(), nullable=False)])
    table = pa.Table.from_pandas(df)
    assert table.schema.field('a').nullable is True
    table = pa.Table.from_pandas(df, schema=schema)
    assert table.schema.field('a').nullable is False