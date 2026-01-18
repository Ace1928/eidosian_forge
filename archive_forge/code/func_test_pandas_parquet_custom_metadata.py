import io
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_pandas_parquet_custom_metadata(tempdir):
    df = alltypes_sample(size=10000)
    filename = tempdir / 'pandas_roundtrip.parquet'
    arrow_table = pa.Table.from_pandas(df)
    assert b'pandas' in arrow_table.schema.metadata
    _write_table(arrow_table, filename)
    metadata = pq.read_metadata(filename).metadata
    assert b'pandas' in metadata
    js = json.loads(metadata[b'pandas'].decode('utf8'))
    assert js['index_columns'] == [{'kind': 'range', 'name': None, 'start': 0, 'stop': 10000, 'step': 1}]