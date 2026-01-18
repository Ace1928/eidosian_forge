import os
import random
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _test_dataframe
from pyarrow.tests.parquet.test_dataset import (
from pyarrow.util import guid
def test_orphaned_file(self):
    hdfs = hdfs_test_client()
    file_path = self._make_test_file(hdfs, 'orphaned_file_test', 'fname', b'foobarbaz')
    f = hdfs.open(file_path)
    hdfs = None
    f = None