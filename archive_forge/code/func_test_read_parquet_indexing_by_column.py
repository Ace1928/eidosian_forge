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
def test_read_parquet_indexing_by_column(self, tmp_path, engine, make_parquet_file):
    nrows = MinPartitionSize.get() + 1
    unique_filename = get_unique_filename(extension='parquet', data_dir=tmp_path)
    make_parquet_file(filename=unique_filename, nrows=nrows)
    parquet_df = pd.read_parquet(unique_filename, engine=engine)
    for col in parquet_df.columns:
        parquet_df[col]