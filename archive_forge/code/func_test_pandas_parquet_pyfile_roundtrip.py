import io
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_pandas_parquet_pyfile_roundtrip(tempdir):
    filename = tempdir / 'pandas_pyfile_roundtrip.parquet'
    size = 5
    df = pd.DataFrame({'int64': np.arange(size, dtype=np.int64), 'float32': np.arange(size, dtype=np.float32), 'float64': np.arange(size, dtype=np.float64), 'bool': np.random.randn(size) > 0, 'strings': ['foo', 'bar', None, 'baz', 'qux']})
    arrow_table = pa.Table.from_pandas(df)
    with filename.open('wb') as f:
        _write_table(arrow_table, f, version='2.6')
    data = io.BytesIO(filename.read_bytes())
    table_read = _read_table(data)
    df_read = table_read.to_pandas()
    tm.assert_frame_equal(df, df_read)