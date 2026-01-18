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
@pytest.mark.parametrize('float_precision', [None, 'high', 'legacy', 'round_trip'])
def test_python_engine_float_precision_except(self, float_precision):
    expected_exception = None
    if float_precision is not None:
        expected_exception = ValueError("The 'float_precision' option is not supported with the 'python' engine")
    eval_io(fn_name='read_csv', filepath_or_buffer=pytest.csvs_names['test_read_csv_regular'], engine='python', float_precision=float_precision, expected_exception=expected_exception)