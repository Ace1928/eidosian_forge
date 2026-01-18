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
def test_to_pandas_deduplicate_date_time():
    nunique = 100
    repeats = 10
    unique_values = list(range(nunique))
    cases = [('int32', 'date32', {'date_as_object': True}), ('int64', 'date64', {'date_as_object': True}), ('int32', 'time32[ms]', {}), ('int64', 'time64[us]', {})]
    for raw_type, array_type, pandas_options in cases:
        raw_arr = pa.array(unique_values * repeats, type=raw_type)
        casted_arr = raw_arr.cast(array_type)
        _assert_nunique(casted_arr.to_pandas(**pandas_options), nunique)
        _assert_nunique(casted_arr.to_pandas(deduplicate_objects=False, **pandas_options), len(casted_arr))