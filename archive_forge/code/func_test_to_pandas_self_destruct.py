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
def test_to_pandas_self_destruct():
    K = 50

    def _make_table():
        return pa.table([pa.array(np.random.randn(10000)[::2]) for i in range(K)], ['f{}'.format(i) for i in range(K)])
    t = _make_table()
    _check_to_pandas_memory_unchanged(t, split_blocks=True, self_destruct=True)
    t = _make_table()
    _check_to_pandas_memory_unchanged(t, self_destruct=True)