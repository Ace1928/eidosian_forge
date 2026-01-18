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
@pytest.mark.filterwarnings('ignore:Sparse:FutureWarning')
@pytest.mark.filterwarnings('ignore:DataFrame.to_sparse:FutureWarning')
def test_sparse_dataframe(version):
    if not pa.pandas_compat._pandas_api.has_sparse:
        pytest.skip('version of pandas does not support SparseDataFrame')
    data = {'A': [0, 1, 2], 'B': [1, 0, 1]}
    df = pd.DataFrame(data).to_sparse(fill_value=1)
    expected = df.to_dense()
    _check_pandas_roundtrip(df, expected, version=version)