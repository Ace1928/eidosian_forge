import contextlib
import datetime
import os
import pathlib
import posixpath
import sys
import tempfile
import textwrap
import threading
import time
from shutil import copytree
from urllib.parse import quote
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.fs as fs
import pyarrow.json
from pyarrow.tests.util import (FSProtocolClass, ProxyHandler,
def test_union_dataset_from_other_datasets(tempdir, multisourcefs):
    child1 = ds.dataset('/plain', filesystem=multisourcefs, format='parquet')
    child2 = ds.dataset('/schema', filesystem=multisourcefs, format='parquet', partitioning=['week', 'color'])
    child3 = ds.dataset('/hive', filesystem=multisourcefs, format='parquet', partitioning='hive')
    assert child1.schema != child2.schema != child3.schema
    assembled = ds.dataset([child1, child2, child3])
    assert isinstance(assembled, ds.UnionDataset)
    msg = 'cannot pass any additional arguments'
    with pytest.raises(ValueError, match=msg):
        ds.dataset([child1, child2], filesystem=multisourcefs)
    expected_schema = pa.schema([('date', pa.date32()), ('index', pa.int64()), ('value', pa.float64()), ('color', pa.string()), ('week', pa.int32()), ('year', pa.int32()), ('month', pa.int32())])
    assert assembled.schema.equals(expected_schema)
    assert assembled.to_table().schema.equals(expected_schema)
    assembled = ds.dataset([child1, child3])
    expected_schema = pa.schema([('date', pa.date32()), ('index', pa.int64()), ('value', pa.float64()), ('color', pa.string()), ('year', pa.int32()), ('month', pa.int32())])
    assert assembled.schema.equals(expected_schema)
    assert assembled.to_table().schema.equals(expected_schema)
    expected_schema = pa.schema([('month', pa.int32()), ('color', pa.string()), ('date', pa.date32())])
    assembled = ds.dataset([child1, child3], schema=expected_schema)
    assert assembled.to_table().schema.equals(expected_schema)
    expected_schema = pa.schema([('month', pa.int32()), ('color', pa.string()), ('unknown', pa.string())])
    assembled = ds.dataset([child1, child3], schema=expected_schema)
    assert assembled.to_table().schema.equals(expected_schema)
    table = pa.table([range(9), [0.0] * 4 + [1.0] * 5, 'abcdefghj'], names=['date', 'value', 'index'])
    _, path = _create_single_file(tempdir, table=table)
    child4 = ds.dataset(path)
    with pytest.raises(pa.ArrowTypeError, match='Unable to merge'):
        ds.dataset([child1, child4])