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
def _create_metadata_file(root_path):
    parquet_paths = list(sorted(root_path.rglob('*.parquet')))
    schema = pq.ParquetFile(parquet_paths[0]).schema.to_arrow_schema()
    metadata_collector = []
    for path in parquet_paths:
        metadata = pq.ParquetFile(path).metadata
        metadata.set_file_path(str(path.relative_to(root_path)))
        metadata_collector.append(metadata)
    metadata_path = root_path / '_metadata'
    pq.write_metadata(schema, metadata_path, metadata_collector=metadata_collector)
    return metadata_path