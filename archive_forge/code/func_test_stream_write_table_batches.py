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
def test_stream_write_table_batches(stream_fixture):
    df = pd.DataFrame({'one': np.random.randn(20)})
    b1 = pa.RecordBatch.from_pandas(df[:10], preserve_index=False)
    b2 = pa.RecordBatch.from_pandas(df, preserve_index=False)
    table = pa.Table.from_batches([b1, b2, b1])
    with stream_fixture._get_writer(stream_fixture.sink, table.schema) as wr:
        wr.write_table(table, max_chunksize=15)
    batches = list(pa.ipc.open_stream(stream_fixture.get_source()))
    assert list(map(len, batches)) == [10, 15, 5, 10]
    result_table = pa.Table.from_batches(batches)
    assert_frame_equal(result_table.to_pandas(), pd.concat([df[:10], df, df[:10]], ignore_index=True))