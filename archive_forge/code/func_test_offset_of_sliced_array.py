from datetime import datetime as dt
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
import pyarrow.interchange as pi
from pyarrow.interchange.column import (
from pyarrow.interchange.from_dataframe import _from_dataframe
def test_offset_of_sliced_array():
    arr = pa.array([1, 2, 3, 4])
    arr_sliced = arr.slice(2, 2)
    table = pa.table([arr], names=['arr'])
    table_sliced = pa.table([arr_sliced], names=['arr_sliced'])
    col = table_sliced.__dataframe__().get_column(0)
    assert col.offset == 2
    result = _from_dataframe(table_sliced.__dataframe__())
    assert table_sliced.equals(result)
    assert not table.equals(result)