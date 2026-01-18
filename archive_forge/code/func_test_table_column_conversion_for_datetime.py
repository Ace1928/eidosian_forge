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
def test_table_column_conversion_for_datetime():
    series = pd.Series(pd.date_range('2012', periods=2, tz='Europe/Brussels'), name='datetime_column')
    table = pa.table({'datetime_column': pa.array(series)})
    table_col = table.column('datetime_column')
    result = table_col.to_pandas()
    assert result.name == 'datetime_column'
    tm.assert_series_equal(result, series)