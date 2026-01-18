import io
import os
import sys
import tempfile
import pytest
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
from pyarrow.feather import (read_feather, write_feather, read_table,
@pytest.mark.pandas
@pytest.mark.xfail(reason='not supported', raises=TypeError)
def test_timedelta_with_nulls_v1():
    df = pd.DataFrame({'test': [pd.Timedelta('1 day'), None, pd.Timedelta('3 day')]})
    _check_pandas_roundtrip(df, version=1)