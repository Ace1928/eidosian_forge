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
def test_roundtrip_nested_map_table_with_pydicts():
    schema = pa.schema([pa.field('a', pa.list_(pa.map_(pa.int8(), pa.struct([pa.field('b', pa.binary())]))))])
    table = pa.table([[[[(1, None)]], None, [[(2, {'b': b'abc'})], [(3, {'b': None}), (4, {'b': b'def'})]]]], schema=schema)
    expected_default_df = pd.DataFrame({'a': [[[(1, None)]], None, [[(2, {'b': b'abc'})], [(3, {'b': None}), (4, {'b': b'def'})]]]})
    expected_as_pydicts_df = pd.DataFrame({'a': [[{1: None}], None, [{2: {'b': b'abc'}}, {3: {'b': None}, 4: {'b': b'def'}}]]})
    default_df = table.to_pandas()
    as_pydicts_df = table.to_pandas(maps_as_pydicts='strict')
    tm.assert_frame_equal(default_df, expected_default_df)
    tm.assert_frame_equal(as_pydicts_df, expected_as_pydicts_df)
    table_default_roundtrip = pa.Table.from_pandas(default_df, schema=schema)
    assert table.equals(table_default_roundtrip)
    table_as_pydicts_roundtrip = pa.Table.from_pandas(as_pydicts_df, schema=schema)
    assert table.equals(table_as_pydicts_roundtrip)