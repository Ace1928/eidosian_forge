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
def test_table_str_to_categorical_with_na(self):
    values = [None, 'a', 'b', np.nan]
    df = pd.DataFrame({'strings': values})
    field = pa.field('strings', pa.string())
    schema = pa.schema([field])
    table = pa.Table.from_pandas(df, schema=schema)
    result = table.to_pandas(strings_to_categorical=True)
    expected = pd.DataFrame({'strings': pd.Categorical(values)})
    tm.assert_frame_equal(result, expected, check_dtype=True)
    with pytest.raises(pa.ArrowInvalid):
        table.to_pandas(strings_to_categorical=True, zero_copy_only=True)