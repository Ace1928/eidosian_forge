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
@pytest.mark.parametrize('filters', [None, [('col1', '<=', 1000000)], [('col1', '<=', 75), ('col2', '>=', 35)]])
@pytest.mark.parametrize('range_index_start', [0, 5000])
@pytest.mark.parametrize('range_index_step', [1, 10])
@pytest.mark.parametrize('range_index_name', [None, 'my_index'])
@pytest.mark.parametrize('row_group_size', [None, 20])
def test_read_parquet_directory_range_index(self, engine, make_parquet_dir, filters, range_index_start, range_index_step, range_index_name, row_group_size):
    self._test_read_parquet_directory(engine=engine, make_parquet_dir=make_parquet_dir, columns=None, filters=filters, range_index_start=range_index_start, range_index_step=range_index_step, range_index_name=range_index_name, row_group_size=row_group_size, rows_per_file=[250] + [0] * 10 + [25] * 10)