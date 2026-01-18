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
@pytest.mark.parquet
def test_parquet_dataset_factory_order(tempdir):
    metadatas = []
    for i in range(10):
        table = pa.table({'f1': list(range(i * 10, (i + 1) * 10))})
        table_path = tempdir / f'{i}.parquet'
        pq.write_table(table, table_path, metadata_collector=metadatas)
        metadatas[-1].set_file_path(f'{i}.parquet')
    metadata_path = str(tempdir / '_metadata')
    pq.write_metadata(table.schema, metadata_path, metadatas)
    dataset = ds.parquet_dataset(metadata_path)
    scanned_table = dataset.to_table()
    scanned_col = scanned_table.column('f1').to_pylist()
    assert scanned_col == list(range(0, 100))