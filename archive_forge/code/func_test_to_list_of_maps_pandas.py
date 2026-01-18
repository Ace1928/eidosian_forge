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
def test_to_list_of_maps_pandas(self):
    if Version(np.__version__) >= Version('1.25.0.dev0') and Version(pd.__version__) < Version('2.0.0'):
        pytest.skip('Regression in pandas with numpy 1.25')
    data = [[[('foo', ['a', 'b']), ('bar', ['c', 'd'])]], [[('baz', []), ('qux', None), ('quux', [None, 'e'])], [('quz', ['f', 'g'])]]]
    arr = pa.array(data, pa.list_(pa.map_(pa.utf8(), pa.list_(pa.utf8()))))
    series = arr.to_pandas()
    expected = pd.Series(data)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'elementwise comparison failed', DeprecationWarning)
        tm.assert_series_equal(series, expected)