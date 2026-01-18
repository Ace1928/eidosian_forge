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
def test_platform_numpy_integers(version):
    data = {}
    numpy_dtypes = ['longlong']
    num_values = 100
    for dtype in numpy_dtypes:
        values = np.random.randint(0, 100, size=num_values)
        data[dtype] = values.astype(dtype)
    df = pd.DataFrame(data)
    _check_pandas_roundtrip(df, version=version)