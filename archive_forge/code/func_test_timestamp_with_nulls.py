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
def test_timestamp_with_nulls(version):
    df = pd.DataFrame({'test': [pd.Timestamp(2016, 1, 1), None, pd.Timestamp(2016, 1, 3)]})
    df['with_tz'] = df.test.dt.tz_localize('utc')
    _check_pandas_roundtrip(df, version=version)