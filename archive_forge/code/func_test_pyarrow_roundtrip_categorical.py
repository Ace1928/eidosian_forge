from datetime import datetime as dt
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
import pyarrow.interchange as pi
from pyarrow.interchange.column import (
from pyarrow.interchange.from_dataframe import _from_dataframe
@pytest.mark.parametrize('offset, length', [(0, 10), (0, 2), (7, 3), (2, 1)])
def test_pyarrow_roundtrip_categorical(offset, length):
    arr = ['Mon', 'Tue', 'Mon', 'Wed', 'Mon', 'Thu', 'Fri', None, 'Sun']
    table = pa.table({'weekday': pa.array(arr).dictionary_encode()})
    table = table.slice(offset, length)
    result = _from_dataframe(table.__dataframe__())
    assert table.equals(result)
    table_protocol = table.__dataframe__()
    result_protocol = result.__dataframe__()
    assert table_protocol.num_columns() == result_protocol.num_columns()
    assert table_protocol.num_rows() == result_protocol.num_rows()
    assert table_protocol.num_chunks() == result_protocol.num_chunks()
    assert table_protocol.column_names() == result_protocol.column_names()
    col_table = table_protocol.get_column(0)
    col_result = result_protocol.get_column(0)
    assert col_result.dtype[0] == DtypeKind.CATEGORICAL
    assert col_result.dtype[0] == col_table.dtype[0]
    assert col_result.size() == col_table.size()
    assert col_result.offset == col_table.offset
    desc_cat_table = col_table.describe_categorical
    desc_cat_result = col_result.describe_categorical
    assert desc_cat_table['is_ordered'] == desc_cat_result['is_ordered']
    assert desc_cat_table['is_dictionary'] == desc_cat_result['is_dictionary']
    assert isinstance(desc_cat_result['categories']._col, pa.Array)