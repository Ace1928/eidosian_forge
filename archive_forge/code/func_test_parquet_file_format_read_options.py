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
def test_parquet_file_format_read_options():
    pff1 = ds.ParquetFileFormat()
    pff2 = ds.ParquetFileFormat(dictionary_columns={'a'})
    pff3 = ds.ParquetFileFormat(coerce_int96_timestamp_unit='s')
    assert pff1.read_options == ds.ParquetReadOptions()
    assert pff2.read_options == ds.ParquetReadOptions(dictionary_columns=['a'])
    assert pff3.read_options == ds.ParquetReadOptions(coerce_int96_timestamp_unit='s')