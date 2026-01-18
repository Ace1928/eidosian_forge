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
def test_safe_cast_from_float_with_nans_to_int():
    values = pd.Series([1, 2, None, 4])
    arr = pa.Array.from_pandas(values, type=pa.int32(), safe=True)
    expected = pa.array([1, 2, None, 4], type=pa.int32())
    assert arr.equals(expected)