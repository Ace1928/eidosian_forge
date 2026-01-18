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
@pytest.mark.parametrize('mask', [None, np.array([True, False, False, True, False, False])])
def test_pandas_datetime_to_date64(self, mask):
    s = pd.to_datetime(['2018-05-10T00:00:00', '2018-05-11T00:00:00', '2018-05-12T00:00:00', '2018-05-10T10:24:01', '2018-05-11T10:24:01', '2018-05-12T10:24:01'])
    arr = pa.Array.from_pandas(s, type=pa.date64(), mask=mask)
    data = np.array([date(2018, 5, 10), date(2018, 5, 11), date(2018, 5, 12), date(2018, 5, 10), date(2018, 5, 11), date(2018, 5, 12)])
    expected = pa.array(data, mask=mask, type=pa.date64())
    assert arr.equals(expected)