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
def test_parquet_write_disable_statistics(tempdir):
    table = pa.Table.from_pydict(OrderedDict([('a', pa.array([1, 2, 3])), ('b', pa.array(['a', 'b', 'c']))]))
    _write_table(table, tempdir / 'data.parquet')
    meta = pq.read_metadata(tempdir / 'data.parquet')
    for col in [0, 1]:
        cc = meta.row_group(0).column(col)
        assert cc.is_stats_set is True
        assert cc.statistics is not None
    _write_table(table, tempdir / 'data2.parquet', write_statistics=False)
    meta = pq.read_metadata(tempdir / 'data2.parquet')
    for col in [0, 1]:
        cc = meta.row_group(0).column(col)
        assert cc.is_stats_set is False
        assert cc.statistics is None
    _write_table(table, tempdir / 'data3.parquet', write_statistics=['a'])
    meta = pq.read_metadata(tempdir / 'data3.parquet')
    cc_a = meta.row_group(0).column(0)
    cc_b = meta.row_group(0).column(1)
    assert cc_a.is_stats_set is True
    assert cc_b.is_stats_set is False
    assert cc_a.statistics is not None
    assert cc_b.statistics is None