from datetime import datetime as dt
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
import pyarrow.interchange as pi
from pyarrow.interchange.column import (
from pyarrow.interchange.from_dataframe import _from_dataframe
@pytest.mark.pandas
@pytest.mark.parametrize('uint', [pa.uint8(), pa.uint16(), pa.uint32()])
@pytest.mark.parametrize('int', [pa.int8(), pa.int16(), pa.int32(), pa.int64()])
@pytest.mark.parametrize('float, np_float', [(pa.float32(), np.float32), (pa.float64(), np.float64)])
def test_pandas_roundtrip(uint, int, float, np_float):
    if Version(pd.__version__) < Version('1.5.0'):
        pytest.skip('__dataframe__ added to pandas in 1.5.0')
    arr = [1, 2, 3]
    table = pa.table({'a': pa.array(arr, type=uint), 'b': pa.array(arr, type=int), 'c': pa.array(np.array(arr, dtype=np_float), type=float), 'd': [True, False, True]})
    from pandas.api.interchange import from_dataframe as pandas_from_dataframe
    pandas_df = pandas_from_dataframe(table)
    result = pi.from_dataframe(pandas_df)
    assert table.equals(result)
    table_protocol = table.__dataframe__()
    result_protocol = result.__dataframe__()
    assert table_protocol.num_columns() == result_protocol.num_columns()
    assert table_protocol.num_rows() == result_protocol.num_rows()
    assert table_protocol.num_chunks() == result_protocol.num_chunks()
    assert table_protocol.column_names() == result_protocol.column_names()