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
def test_stream_write_dispatch(stream_fixture):
    df = pd.DataFrame({'one': np.random.randn(5), 'two': pd.Categorical(['foo', np.nan, 'bar', 'foo', 'foo'], categories=['foo', 'bar'], ordered=True)})
    table = pa.Table.from_pandas(df, preserve_index=False)
    batch = pa.RecordBatch.from_pandas(df, preserve_index=False)
    with stream_fixture._get_writer(stream_fixture.sink, table.schema) as wr:
        wr.write(table)
        wr.write(batch)
    table = pa.ipc.open_stream(pa.BufferReader(stream_fixture.get_source())).read_all()
    assert_frame_equal(table.to_pandas(), pd.concat([df, df], ignore_index=True))