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
@pytest.mark.parametrize('index', ['a', ['a', 'b']])
def test_to_pandas_types_mapper_index(index):
    if Version(pd.__version__) < Version('1.5.0'):
        pytest.skip('ArrowDtype missing')
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}, dtype=pd.ArrowDtype(pa.int64())).set_index(index)
    expected = df.copy()
    table = pa.table(df)
    result = table.to_pandas(types_mapper=pd.ArrowDtype)
    tm.assert_frame_equal(result, expected)