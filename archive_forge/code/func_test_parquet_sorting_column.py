import datetime
import decimal
from collections import OrderedDict
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip, make_sample_file
from pyarrow.fs import LocalFileSystem
from pyarrow.tests import util
def test_parquet_sorting_column():
    sorting_col = pq.SortingColumn(10)
    assert sorting_col.column_index == 10
    assert sorting_col.descending is False
    assert sorting_col.nulls_first is False
    sorting_col = pq.SortingColumn(0, descending=True, nulls_first=True)
    assert sorting_col.column_index == 0
    assert sorting_col.descending is True
    assert sorting_col.nulls_first is True
    schema = pa.schema([('a', pa.int64()), ('b', pa.int64())])
    sorting_cols = (pq.SortingColumn(1, descending=True), pq.SortingColumn(0, descending=False))
    sort_order, null_placement = pq.SortingColumn.to_ordering(schema, sorting_cols)
    assert sort_order == (('b', 'descending'), ('a', 'ascending'))
    assert null_placement == 'at_end'
    sorting_cols_roundtripped = pq.SortingColumn.from_ordering(schema, sort_order, null_placement)
    assert sorting_cols_roundtripped == sorting_cols
    sorting_cols = pq.SortingColumn.from_ordering(schema, ('a', ('b', 'descending')), null_placement='at_start')
    expected = (pq.SortingColumn(0, descending=False, nulls_first=True), pq.SortingColumn(1, descending=True, nulls_first=True))
    assert sorting_cols == expected
    empty_sorting_cols = pq.SortingColumn.from_ordering(schema, ())
    assert empty_sorting_cols == ()
    assert pq.SortingColumn.to_ordering(schema, ()) == ((), 'at_end')
    with pytest.raises(ValueError):
        pq.SortingColumn.from_ordering(schema, ('a', 'not a valid sort order'))
    with pytest.raises(ValueError, match='inconsistent null placement'):
        sorting_cols = (pq.SortingColumn(1, nulls_first=True), pq.SortingColumn(0, nulls_first=False))
        pq.SortingColumn.to_ordering(schema, sorting_cols)