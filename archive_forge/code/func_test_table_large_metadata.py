import datetime
import decimal
from collections import OrderedDict
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip, make_sample_file
from pyarrow.fs import LocalFileSystem
from pyarrow.tests import util
def test_table_large_metadata():
    my_schema = pa.schema([pa.field('f0', 'double')], metadata={'large': 'x' * 10000000})
    table = pa.table([np.arange(10)], schema=my_schema)
    _check_roundtrip(table)