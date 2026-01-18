from __future__ import annotations
import os
import sys
from array import array
import pytest
from dask.multiprocessing import get_context
from dask.sizeof import sizeof
from dask.utils import funcname
@requires_pandas
def test_pyarrow_table():
    pa = pytest.importorskip('pyarrow')
    df = pd.DataFrame({'x': [1, 2, 3], 'y': ['a' * 100, 'b' * 100, 'c' * 100]}, index=[10, 20, 30])
    table = pa.Table.from_pandas(df)
    assert sizeof(table) > sizeof(table.schema.metadata)
    assert isinstance(sizeof(table), int)
    assert isinstance(sizeof(table.columns[0]), int)
    assert isinstance(sizeof(table.columns[1]), int)
    assert isinstance(sizeof(table.columns[2]), int)
    empty = pa.Table.from_pandas(df.head(0))
    assert sizeof(empty) > sizeof(empty.schema.metadata)
    assert sizeof(empty.columns[0]) > 0
    assert sizeof(empty.columns[1]) > 0
    assert sizeof(empty.columns[2]) > 0