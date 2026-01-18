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
def test_v2_set_chunksize():
    df = pd.DataFrame({'A': np.arange(1000)})
    table = pa.table(df)
    buf = io.BytesIO()
    write_feather(table, buf, chunksize=250, version=2)
    result = buf.getvalue()
    ipc_file = pa.ipc.open_file(pa.BufferReader(result))
    assert ipc_file.num_record_batches == 4
    assert len(ipc_file.get_batch(0)) == 250