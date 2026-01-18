from collections import OrderedDict
import io
import warnings
from shutil import copytree
import numpy as np
import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem, FileSystem
from pyarrow.tests import util
from pyarrow.tests.parquet.common import (_check_roundtrip, _roundtrip_table,
def test_write_nested_zero_length_array_chunk_failure():
    cols = OrderedDict(int32=pa.int32(), list_string=pa.list_(pa.string()))
    data = [[], [OrderedDict(int32=1, list_string=('G',))]]
    my_arrays = [pa.array(batch, type=pa.struct(cols)).flatten() for batch in data]
    my_batches = [pa.RecordBatch.from_arrays(batch, schema=pa.schema(cols)) for batch in my_arrays]
    tbl = pa.Table.from_batches(my_batches, pa.schema(cols))
    _check_roundtrip(tbl)