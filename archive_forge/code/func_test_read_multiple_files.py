import datetime
import inspect
import os
import pathlib
import numpy as np
import pytest
import unittest.mock as mock
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem
from pyarrow.tests import util
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_read_multiple_files(tempdir):
    nfiles = 10
    size = 5
    dirpath = tempdir / guid()
    dirpath.mkdir()
    test_data = []
    paths = []
    for i in range(nfiles):
        df = _test_dataframe(size, seed=i)
        df['uint32'] = df['uint32'].astype(np.int64)
        path = dirpath / '{}.parquet'.format(i)
        table = pa.Table.from_pandas(df)
        _write_table(table, path)
        test_data.append(table)
        paths.append(path)
    (dirpath / '_SUCCESS.crc').touch()

    def read_multiple_files(paths, columns=None, use_threads=True, **kwargs):
        dataset = pq.ParquetDataset(paths, **kwargs)
        return dataset.read(columns=columns, use_threads=use_threads)
    result = read_multiple_files(paths)
    expected = pa.concat_tables(test_data)
    assert result.equals(expected)
    to_read = [0, 2, 6, result.num_columns - 1]
    col_names = [result.field(i).name for i in to_read]
    out = pq.read_table(dirpath, columns=col_names)
    expected = pa.Table.from_arrays([result.column(i) for i in to_read], names=col_names, metadata=result.schema.metadata)
    assert out.equals(expected)
    pq.read_table(dirpath, use_threads=True)
    bad_apple = _test_dataframe(size, seed=i).iloc[:, :4]
    bad_apple_path = tempdir / '{}.parquet'.format(guid())
    t = pa.Table.from_pandas(bad_apple)
    _write_table(t, bad_apple_path)