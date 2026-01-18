import sys
import sysconfig
import pytest
import pyarrow as pa
import numpy as np
def test_table_deserialize():
    hbuf, htable, dbuf, dtable = make_table_cuda()
    assert htable.schema == dtable.schema
    assert htable.num_rows == dtable.num_rows
    assert htable.num_columns == dtable.num_columns
    assert hbuf.equals(dbuf.copy_to_host())
    assert htable.equals(pa.ipc.open_stream(dbuf.copy_to_host()).read_all())