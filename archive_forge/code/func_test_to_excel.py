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
@pytest.mark.xfail(strict=False, reason='Flaky test, defaults to pandas')
def test_to_excel(self, tmp_path):
    modin_df, pandas_df = create_test_dfs(TEST_DATA)
    unique_filename_modin = get_unique_filename(extension='xlsx', data_dir=tmp_path)
    unique_filename_pandas = get_unique_filename(extension='xlsx', data_dir=tmp_path)
    modin_writer = pandas.ExcelWriter(unique_filename_modin)
    pandas_writer = pandas.ExcelWriter(unique_filename_pandas)
    modin_df.to_excel(modin_writer)
    pandas_df.to_excel(pandas_writer)
    modin_writer.save()
    pandas_writer.save()
    assert assert_files_eq(unique_filename_modin, unique_filename_pandas)