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
def _create_parquet_dataset_simple(root_path):
    """
    Creates a simple (flat files, no nested partitioning) Parquet dataset
    """
    metadata_collector = []
    for i in range(4):
        table = pa.table({'f1': [i] * 10, 'f2': np.random.randn(10)})
        pq.write_to_dataset(table, str(root_path), metadata_collector=metadata_collector)
    metadata_path = str(root_path / '_metadata')
    pq.write_metadata(table.schema, metadata_path, metadata_collector=metadata_collector)
    return (metadata_path, table)