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
@pytest.mark.orc
def test_orc_format(tempdir, dataset_reader):
    from pyarrow import orc
    table = pa.table({'a': pa.array([1, 2, 3], type='int8'), 'b': pa.array([0.1, 0.2, 0.3], type='float64')})
    path = str(tempdir / 'test.orc')
    orc.write_table(table, path)
    dataset = ds.dataset(path, format=ds.OrcFileFormat())
    fragments = list(dataset.get_fragments())
    assert isinstance(fragments[0], ds.FileFragment)
    result = dataset_reader.to_table(dataset)
    result.validate(full=True)
    assert result.equals(table)
    assert_dataset_fragment_convenience_methods(dataset)
    dataset = ds.dataset(path, format='orc')
    result = dataset_reader.to_table(dataset)
    result.validate(full=True)
    assert result.equals(table)
    result = dataset_reader.to_table(dataset, columns=['b'])
    result.validate(full=True)
    assert result.equals(table.select(['b']))
    result = dataset_reader.to_table(dataset, columns={'b2': ds.field('b') * 2})
    result.validate(full=True)
    assert result.equals(pa.table({'b2': pa.array([0.2, 0.4, 0.6], type='float64')}))
    assert dataset_reader.count_rows(dataset) == 3
    assert dataset_reader.count_rows(dataset, filter=ds.field('a') > 2) == 1