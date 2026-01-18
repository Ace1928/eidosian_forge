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
@pytest.mark.parametrize('index', [False, True])
def test_read_parquet_6855(self, tmp_path, engine, index):
    if engine == 'fastparquet':
        pytest.skip("integer columns aren't supported")
    test_df = pandas.DataFrame(np.random.rand(10 ** 2, 10))
    path = tmp_path / 'data'
    path.mkdir()
    file_name = 'issue6855.parquet'
    test_df.to_parquet(path / file_name, index=index, engine=engine)
    read_df = pd.read_parquet(path / file_name, engine=engine)
    if not index:
        read_df.columns = pandas.Index(read_df.columns).astype('int64').to_list()
    df_equals(test_df, read_df)