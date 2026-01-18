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
@pytest.mark.parametrize('columns', [None, ['col1']])
@pytest.mark.parametrize('filters', [None, [('col1', '<=', 3215), ('col2', '>=', 35)]])
@pytest.mark.parametrize('row_group_size', [None, 100, 1000, 10000])
@pytest.mark.parametrize('rows_per_file', [[1000] * 40, [0, 0, 40000], [10000, 10000] + [100] * 200])
@pytest.mark.exclude_in_sanity
def test_read_parquet_directory(self, engine, make_parquet_dir, columns, filters, row_group_size, rows_per_file):
    self._test_read_parquet_directory(engine=engine, make_parquet_dir=make_parquet_dir, columns=columns, filters=filters, range_index_start=0, range_index_step=1, range_index_name=None, row_group_size=row_group_size, rows_per_file=rows_per_file)