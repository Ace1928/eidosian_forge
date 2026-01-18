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
def test_parquet_sorting_column_nested():
    schema = pa.schema({'a': pa.struct([('x', pa.int64()), ('y', pa.int64())]), 'b': pa.int64()})
    sorting_columns = [pq.SortingColumn(0, descending=True), pq.SortingColumn(2, descending=False)]
    sort_order, null_placement = pq.SortingColumn.to_ordering(schema, sorting_columns)
    assert null_placement == 'at_end'
    assert len(sort_order) == 2
    assert sort_order[0] == ('a.x', 'descending')
    assert sort_order[1] == ('b', 'ascending')