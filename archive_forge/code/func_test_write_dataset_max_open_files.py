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
def test_write_dataset_max_open_files(tempdir):
    directory = tempdir / 'ds'
    file_format = 'parquet'
    partition_column_id = 1
    column_names = ['c1', 'c2']
    record_batch_1 = pa.record_batch(data=[[1, 2, 3, 4, 0, 10], ['a', 'b', 'c', 'd', 'e', 'a']], names=column_names)
    record_batch_2 = pa.record_batch(data=[[5, 6, 7, 8, 0, 1], ['a', 'b', 'c', 'd', 'e', 'c']], names=column_names)
    record_batch_3 = pa.record_batch(data=[[9, 10, 11, 12, 0, 1], ['a', 'b', 'c', 'd', 'e', 'd']], names=column_names)
    record_batch_4 = pa.record_batch(data=[[13, 14, 15, 16, 0, 1], ['a', 'b', 'c', 'd', 'e', 'b']], names=column_names)
    table = pa.Table.from_batches([record_batch_1, record_batch_2, record_batch_3, record_batch_4])
    partitioning = ds.partitioning(pa.schema([(column_names[partition_column_id], pa.string())]), flavor='hive')
    data_source_1 = directory / 'default'
    ds.write_dataset(data=table, base_dir=data_source_1, partitioning=partitioning, format=file_format)

    def _get_compare_pair(data_source, record_batch, file_format, col_id):
        num_of_files_generated = _get_num_of_files_generated(base_directory=data_source, file_format=file_format)
        number_of_partitions = len(pa.compute.unique(record_batch[col_id]))
        return (num_of_files_generated, number_of_partitions)
    num_of_files_generated, number_of_partitions = _get_compare_pair(data_source_1, record_batch_1, file_format, partition_column_id)
    assert num_of_files_generated == number_of_partitions
    data_source_2 = directory / 'max_1'
    max_open_files = 3
    ds.write_dataset(data=table, base_dir=data_source_2, partitioning=partitioning, format=file_format, max_open_files=max_open_files, use_threads=False)
    num_of_files_generated, number_of_partitions = _get_compare_pair(data_source_2, record_batch_1, file_format, partition_column_id)
    assert num_of_files_generated > number_of_partitions