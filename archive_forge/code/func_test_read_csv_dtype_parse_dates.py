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
@pytest.mark.parametrize('date', ['2023-01-01 00:00:01.000000000', '2023'])
@pytest.mark.parametrize('dtype', [None, 'str', {'id': 'int64'}])
@pytest.mark.parametrize('parse_dates', [None, [], ['date'], [1]])
def test_read_csv_dtype_parse_dates(self, date, dtype, parse_dates):
    with ensure_clean('.csv') as filename:
        with open(filename, 'w') as file:
            file.write(f'id,date\n1,{date}')
        eval_io(fn_name='read_csv', filepath_or_buffer=filename, dtype=dtype, parse_dates=parse_dates)