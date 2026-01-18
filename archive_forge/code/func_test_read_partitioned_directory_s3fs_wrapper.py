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
@pytest.mark.pandas
@pytest.mark.s3
def test_read_partitioned_directory_s3fs_wrapper(s3_example_s3fs):
    import s3fs
    from pyarrow.filesystem import S3FSWrapper
    if Version(s3fs.__version__) >= Version('0.5'):
        pytest.skip('S3FSWrapper no longer working for s3fs 0.5+')
    fs, path = s3_example_s3fs
    with pytest.warns(FutureWarning):
        wrapper = S3FSWrapper(fs)
    _partition_test_for_filesystem(wrapper, path)
    dataset = pq.ParquetDataset(path, filesystem=fs)
    dataset.read()