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
def test_write_dataset_min_rows_per_group(tempdir):
    directory = tempdir / 'ds'
    min_rows_per_group = 6
    max_rows_per_group = 8
    num_of_columns = 2
    record_sizes = [5, 5, 5, 5, 5, 4, 4, 4, 4, 4]
    record_batches = [_generate_data_and_columns(num_of_columns, num_of_records) for num_of_records in record_sizes]
    data_source = directory / 'min_rows_group'
    ds.write_dataset(data=record_batches, base_dir=data_source, min_rows_per_group=min_rows_per_group, max_rows_per_group=max_rows_per_group, format='parquet')
    files_in_dir = os.listdir(data_source)
    for _, f_file in enumerate(files_in_dir):
        f_path = data_source / str(f_file)
        dataset = ds.dataset(f_path, format='parquet')
        table = dataset.to_table()
        batches = table.to_batches()
        for id, batch in enumerate(batches):
            rows_per_batch = batch.num_rows
            if id < len(batches) - 1:
                assert rows_per_batch >= min_rows_per_group and rows_per_batch <= max_rows_per_group
            else:
                assert rows_per_batch <= max_rows_per_group