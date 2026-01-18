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
def test_to_parquet_keep_index(self, tmp_path, engine):
    data = {'c0': [0, 1] * 1000, 'c1': [2, 3] * 1000}
    modin_df, pandas_df = create_test_dfs(data)
    modin_df.index.name = 'foo'
    pandas_df.index.name = 'foo'
    parquet_eval_to_file(tmp_path, modin_obj=modin_df, pandas_obj=pandas_df, fn='to_parquet', extension='parquet', index=True, engine=engine)