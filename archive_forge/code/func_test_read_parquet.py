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
@pytest.mark.parametrize('row_group_size', [None, 100, 1000, 10000])
@pytest.mark.parametrize('path_type', [Path, str])
def test_read_parquet(self, engine, make_parquet_file, columns, row_group_size, path_type):
    self._test_read_parquet(engine=engine, make_parquet_file=make_parquet_file, columns=columns, filters=None, row_group_size=row_group_size, path_type=path_type)