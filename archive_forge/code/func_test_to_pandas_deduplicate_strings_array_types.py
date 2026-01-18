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
def test_to_pandas_deduplicate_strings_array_types():
    nunique = 100
    repeats = 10
    values = _generate_dedup_example(nunique, repeats)
    for arr in [pa.array(values, type=pa.binary()), pa.array(values, type=pa.utf8()), pa.chunked_array([values, values])]:
        _assert_nunique(arr.to_pandas(), nunique)
        _assert_nunique(arr.to_pandas(deduplicate_objects=False), len(arr))