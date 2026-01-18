import io
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_pandas_can_write_nested_data():
    data = {'agg_col': [{'page_type': 1}, {'record_type': 1}, {'non_consecutive_home': 0}], 'uid_first': '1001'}
    df = pd.DataFrame(data=data)
    arrow_table = pa.Table.from_pandas(df)
    imos = pa.BufferOutputStream()
    _write_table(arrow_table, imos)