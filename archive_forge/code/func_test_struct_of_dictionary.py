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
def test_struct_of_dictionary(self):
    names = ['ints', 'strs']
    children = [pa.array([456, 789, 456]).dictionary_encode(), pa.array(['foo', 'foo', None]).dictionary_encode()]
    arr = pa.StructArray.from_arrays(children, names=names)
    rows_as_tuples = zip(*(child.to_pylist() for child in children))
    rows_as_dicts = [dict(zip(names, row)) for row in rows_as_tuples]
    expected = pd.Series(rows_as_dicts)
    tm.assert_series_equal(arr.to_pandas(), expected)
    arr = arr.take([0, None, 2])
    expected[1] = None
    tm.assert_series_equal(arr.to_pandas(), expected)