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
@check_file_leaks
def test_read_excel_all_sheets(self, make_excel_file):
    unique_filename = make_excel_file()
    pandas_df = pandas.read_excel(unique_filename, sheet_name=None)
    modin_df = pd.read_excel(unique_filename, sheet_name=None)
    assert isinstance(pandas_df, dict)
    assert isinstance(modin_df, type(pandas_df))
    assert pandas_df.keys() == modin_df.keys()
    for key in pandas_df.keys():
        df_equals(modin_df.get(key), pandas_df.get(key))