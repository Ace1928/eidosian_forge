import contextlib
import csv
import inspect
import os
import sys
import unittest.mock as mock
from collections import defaultdict
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict
import fastparquet
import numpy as np
import pandas
import pandas._libs.lib as lib
import pyarrow as pa
import pyarrow.dataset
import pytest
import sqlalchemy as sa
from packaging import version
from pandas._testing import ensure_clean
from pandas.errors import ParserWarning
from scipy import sparse
from modin.config import (
from modin.db_conn import ModinDatabaseConnection, UnsupportedDatabaseException
from modin.pandas.io import from_arrow, from_dask, from_ray, to_pandas
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from .utils import (
from .utils import test_data as utils_test_data
from .utils import time_parsing_csv_path
from modin.config import NPartitions
@pytest.mark.parametrize('filters', [None, [], [('B', '==', 'a')], [('B', '==', 'a'), ('A', '>=', 5), ('idx', '<=', 30000)]])
def test_read_parquet_pandas_index_partitioned(self, tmp_path, engine, filters):
    pandas_df = pandas.DataFrame({'idx': np.random.randint(0, 100000, size=2000), 'A': np.random.randint(0, 10, size=2000), 'B': ['a', 'b'] * 1000, 'C': ['c'] * 2000})
    unique_filename = get_unique_filename(extension='parquet', data_dir=tmp_path)
    pandas_df.set_index('idx').to_parquet(unique_filename, partition_cols=['A'])
    expected_exception = None
    if filters == [] and engine == 'pyarrow':
        expected_exception = ValueError('Malformed filters')
    eval_io('read_parquet', path=unique_filename, engine=engine, filters=filters, expected_exception=expected_exception)