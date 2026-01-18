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
@pytest.mark.parametrize('filters', [None, [], [('col1', '==', 5)], [('col1', '<=', 215), ('col2', '>=', 35)]])
def test_read_parquet_filters(self, engine, make_parquet_file, filters):
    expected_exception = None
    if filters == [] and engine == 'pyarrow':
        expected_exception = ValueError('Malformed filters')
    self._test_read_parquet(engine=engine, make_parquet_file=make_parquet_file, columns=None, filters=filters, row_group_size=100, path_type=str, expected_exception=expected_exception)