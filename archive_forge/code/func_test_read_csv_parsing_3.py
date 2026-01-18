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
@pytest.mark.parametrize('true_values', [['Yes'], ['Yes', 'true'], None])
@pytest.mark.parametrize('false_values', [['No'], ['No', 'false'], None])
@pytest.mark.parametrize('skipfooter', [0, 10])
@pytest.mark.parametrize('nrows', [35, None])
def test_read_csv_parsing_3(self, true_values, false_values, skipfooter, nrows):
    xfail_case = (false_values or true_values) and Engine.get() != 'Python' and (StorageFormat.get() != 'Hdk')
    if xfail_case:
        pytest.xfail('modin and pandas dataframes differs - issue #2446')
    expected_exception = None
    if skipfooter != 0 and nrows is not None:
        expected_exception = ValueError("'skipfooter' not supported with 'nrows'")
    eval_io(fn_name='read_csv', expected_exception=expected_exception, filepath_or_buffer=pytest.csvs_names['test_read_csv_yes_no'], true_values=true_values, false_values=false_values, skipfooter=skipfooter, nrows=nrows)