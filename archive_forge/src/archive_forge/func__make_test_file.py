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
def _make_test_file(self, hdfs, test_name, test_path, test_data):
    base_path = pjoin(self.tmp_path, test_name)
    hdfs.mkdir(base_path)
    full_path = pjoin(base_path, test_path)
    with hdfs.open(full_path, 'wb') as f:
        f.write(test_data)
    return full_path