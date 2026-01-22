import datetime
import inspect
import os
import pathlib
import numpy as np
import pytest
import unittest.mock as mock
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem
from pyarrow.tests import util
from pyarrow.util import guid
from pyarrow.vendored.version import Version
def _make_example_multifile_dataset(base_path, nfiles=10, file_nrows=5):
    test_data = []
    paths = []
    for i in range(nfiles):
        df = _test_dataframe(file_nrows, seed=i)
        path = base_path / '{}.parquet'.format(i)
        test_data.append(_write_table(df, path))
        paths.append(path)
    return paths