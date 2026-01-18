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
@pytest.mark.lz4
@pytest.mark.snappy
@pytest.mark.zstd
def test_v2_compression_options():
    df = pd.DataFrame({'A': np.arange(1000)})
    cases = [('uncompressed', None), ('lz4', None), ('lz4', 1), ('lz4', 12), ('zstd', 1), ('zstd', 10)]
    for compression, compression_level in cases:
        _check_pandas_roundtrip(df, compression=compression, compression_level=compression_level)
    buf = io.BytesIO()
    with pytest.raises(ValueError, match='Feather V1 files do not support compression option'):
        write_feather(df, buf, compression='lz4', version=1)
    with pytest.raises(ValueError, match='Feather V1 files do not support chunksize option'):
        write_feather(df, buf, chunksize=4096, version=1)
    with pytest.raises(ValueError, match='compression="snappy" not supported'):
        write_feather(df, buf, compression='snappy')