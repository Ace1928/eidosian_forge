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
def test_write_dataset_max_rows_per_group(tempdir):
    directory = tempdir / 'ds'
    max_rows_per_group = 18
    num_of_columns = 2
    num_of_records = 30
    record_batch = _generate_data_and_columns(num_of_columns, num_of_records)
    data_source = directory / 'max_rows_group'
    ds.write_dataset(data=record_batch, base_dir=data_source, max_rows_per_group=max_rows_per_group, format='parquet')
    files_in_dir = os.listdir(data_source)
    batched_data = []
    for f_file in files_in_dir:
        f_path = data_source / str(f_file)
        dataset = ds.dataset(f_path, format='parquet')
        table = dataset.to_table()
        batches = table.to_batches()
        for batch in batches:
            batched_data.append(batch.num_rows)
    assert batched_data == [18, 12]