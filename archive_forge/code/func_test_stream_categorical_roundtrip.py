from collections import UserList
import io
import pathlib
import pytest
import socket
import threading
import weakref
import numpy as np
import pyarrow as pa
from pyarrow.tests.util import changed_environ, invoke_script
@pytest.mark.pandas
def test_stream_categorical_roundtrip(stream_fixture):
    df = pd.DataFrame({'one': np.random.randn(5), 'two': pd.Categorical(['foo', np.nan, 'bar', 'foo', 'foo'], categories=['foo', 'bar'], ordered=True)})
    batch = pa.RecordBatch.from_pandas(df)
    with stream_fixture._get_writer(stream_fixture.sink, batch.schema) as wr:
        wr.write_batch(batch)
    table = pa.ipc.open_stream(pa.BufferReader(stream_fixture.get_source())).read_all()
    assert_frame_equal(table.to_pandas(), df)