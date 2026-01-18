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
def test_parquet_metadata_empty_to_dict(tempdir):
    table = pa.table({'a': pa.array([], type='int64')})
    pq.write_table(table, tempdir / 'data.parquet')
    metadata = pq.read_metadata(tempdir / 'data.parquet')
    metadata_dict = metadata.to_dict()
    assert len(metadata_dict['row_groups']) == 1
    assert len(metadata_dict['row_groups'][0]['columns']) == 1
    assert metadata_dict['row_groups'][0]['columns'][0]['statistics'] is None