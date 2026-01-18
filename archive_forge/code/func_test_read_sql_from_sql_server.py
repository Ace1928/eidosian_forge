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
@pytest.mark.skipif(not TestReadFromSqlServer.get(), reason='Skip the test when the test SQL server is not set up.')
def test_read_sql_from_sql_server(self):
    table_name = 'test_1000x256'
    query = f'SELECT * FROM {table_name}'
    sqlalchemy_connection_string = 'mssql+pymssql://sa:Strong.Pwd-123@0.0.0.0:1433/master'
    pandas_df_to_read = pandas.DataFrame(np.arange(1000 * 256).reshape(1000, 256)).add_prefix('col')
    pandas_df_to_read.to_sql(table_name, sqlalchemy_connection_string, if_exists='replace')
    modin_df = pd.read_sql(query, ModinDatabaseConnection('sqlalchemy', sqlalchemy_connection_string))
    pandas_df = pandas.read_sql(query, sqlalchemy_connection_string)
    df_equals(modin_df, pandas_df)