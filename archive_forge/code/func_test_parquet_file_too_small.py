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
def test_parquet_file_too_small(tempdir):
    path = str(tempdir / 'test.parquet')
    with pytest.raises((pa.ArrowInvalid, OSError), match='size is 0 bytes'):
        with open(path, 'wb') as f:
            pass
        pq.read_table(path)
    with pytest.raises((pa.ArrowInvalid, OSError), match='size is 4 bytes'):
        with open(path, 'wb') as f:
            f.write(b'ffff')
        pq.read_table(path)