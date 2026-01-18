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
def test_write_dataset_write_page_index(tempdir):
    for write_statistics in [True, False]:
        for write_page_index in [True, False]:
            schema = pa.schema([pa.field('x', pa.int64()), pa.field('y', pa.int64())])
            arrays = [[1, 2, 3], [None, 5, None]]
            table = pa.Table.from_arrays(arrays, schema=schema)
            file_format = ds.ParquetFileFormat()
            base_dir = tempdir / f'write_page_index_{write_page_index}'
            ds.write_dataset(table, base_dir, format='parquet', file_options=file_format.make_write_options(write_statistics=write_statistics, write_page_index=write_page_index), existing_data_behavior='overwrite_or_ignore')
            ds1 = ds.dataset(base_dir, format='parquet')
            for file in ds1.files:
                metadata = pq.read_metadata(file)
                cc = metadata.row_group(0).column(0)
                assert cc.has_offset_index is write_page_index
                assert cc.has_column_index is write_page_index & write_statistics