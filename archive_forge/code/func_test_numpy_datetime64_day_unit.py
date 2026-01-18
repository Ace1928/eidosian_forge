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
@pytest.mark.parametrize('dtype', [pa.date32(), pa.date64()])
def test_numpy_datetime64_day_unit(self, dtype):
    datetime64_d = np.array(['2007-07-13', None, '2006-01-15', '2010-08-19'], dtype='datetime64[D]')
    _check_array_from_pandas_roundtrip(datetime64_d, type=dtype)